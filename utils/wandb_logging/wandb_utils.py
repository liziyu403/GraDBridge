import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))  # add utils/ to path
from utils.general import colorstr

try:
    import wandb
except ImportError:
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def wandb_disabled():
    return os.getenv('WANDB_DISABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def wandb_available():
    return (wandb is not None) and (not wandb_disabled())


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return run_id, project, model_artifact_name


def build_wandb_run_name(opt, fallback_name):
    cfg = opt.cfg[0] if isinstance(opt.cfg, (list, tuple)) and opt.cfg else opt.cfg
    cfg_stem = Path(cfg).stem if cfg else str(fallback_name).strip()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{cfg_stem}_{timestamp}"


def check_wandb_resume(opt):
    if not wandb_available():
        return None
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            if opt.global_rank not in [-1, 0]:  # For resuming DDP runs
                run_id, project, model_artifact_name = get_run_info(opt.resume)
                api = wandb.Api()
                artifact = api.artifact(project + '/' + model_artifact_name + ':latest')
                modeldir = artifact.download()
                opt.weights = str(Path(modeldir) / "last.pt")
            return True
    return None

class WandbLogger():
    def __init__(self, opt, name, run_id, data_dict, job_type='Training'):
        self.job_type = job_type
        self.wandb = wandb if wandb_available() else None
        self.wandb_run = None if not self.wandb else self.wandb.run
        self.data_dict = data_dict
        if isinstance(opt.resume, str):  # checks resume from artifact
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                run_id, project, model_artifact_name = get_run_info(opt.resume)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert self.wandb, 'install wandb to resume wandb runs'
                self.wandb_run = self.wandb.init(
                    id=run_id,
                    project=project,
                    resume='allow',
                    dir=str(Path.cwd()),
                )
                opt.resume = model_artifact_name
        elif self.wandb:
            run_name = build_wandb_run_name(opt, name)
            self.wandb_run = self.wandb.init(config=opt,
                                             resume="allow",
                                             project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                             name=run_name,
                                             job_type=job_type,
                                             id=run_id,
                                             dir=str(Path.cwd())) if not self.wandb.run else self.wandb.run
        if self.wandb_run:
            if self.job_type == 'Training':
                if not opt.resume:
                    self.wandb_run.config.opt = vars(opt)
                    self.wandb_run.config.data_dict = data_dict
                self.data_dict = self.setup_training(opt, data_dict)
        else:
            prefix = colorstr('wandb: ')
            print(f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

    def setup_training(self, opt, data_dict):
        self.log_dict, self.current_epoch, self.current_step, self.log_imgs = {}, 0, None, 16  # Logging Constants
        self.bbox_interval = opt.bbox_interval
        self._metrics_defined = False
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp = str(
                    self.weights), config.save_period, config.total_batch_size, config.bbox_interval, config.epochs, \
                                                                                                       config.opt['hyp']
            data_dict = dict(self.wandb_run.config.data_dict)  # eliminates the need for config file to resume
        self.weights = None
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
        self._define_metrics()
        return data_dict

    def _define_metrics(self):
        if not self.wandb_run or self._metrics_defined:
            return
        try:
            wandb.define_metric("train/step")
            wandb.define_metric("train_batch/*", step_metric="train/step")
            wandb.define_metric("lr_batch/*", step_metric="train/step")

            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("metrics/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("lr/*", step_metric="epoch")
            wandb.define_metric("x/*", step_metric="epoch")
        except Exception:
            pass
        self._metrics_defined = True

    def download_model_artifact(self, opt):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX) + ":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            assert epochs_trained < total_epochs, 'training to %g epochs is finished, nothing to resume.' % (
                total_epochs)
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
            'original_url': str(path),
            'epochs_trained': epoch + 1,
            'save period': opt.save_period,
            'project': opt.project,
            'total_epochs': opt.epochs,
            'fitness_score': fitness_score
        })
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        wandb.log_artifact(model_artifact,
                           aliases=['latest', 'epoch ' + str(self.current_epoch), 'best' if best_model else ''])
        print("Saving model artifact on epoch ", epoch + 1)

    def log(self, log_dict, step=None, commit=True):
        if not self.wandb_run:
            return
        if step is None:
            for key, value in log_dict.items():
                self.log_dict[key] = value
            return
        self.current_step = int(step)
        wandb.log(log_dict, step=step, commit=commit)

    def end_epoch(self):
        if self.wandb_run:
            payload = dict(self.log_dict)
            payload.setdefault("epoch", self.current_epoch)
            wandb.log(payload)
            self.log_dict = {}

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                payload = dict(self.log_dict)
                payload.setdefault("epoch", self.current_epoch)
                wandb.log(payload)
            wandb.run.finish()
