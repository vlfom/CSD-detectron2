import detectron2.utils.comm as comm

from detectron2.engine import DefaultTrainer


class BaselineTrainer(DefaultTrainer):
    pass


class CSDTrainer(DefaultTrainer):

    # * __init__ is kept as in DefaultTrainer

    @classmethod
    def build_train_loader(cls, cfg):
        """Overrides default build_train_loader with a custom DataLoader w/ x-flips"""
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)
