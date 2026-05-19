from copy import deepcopy

from default_settings import GeneralSettings, canonical_dataset_name


class PHDDatasetConfig:
    """Dataset-specific presets for the clean MemoTrack entry.

    These presets only tune thresholds and post-processing choices. They do not
    enable ablation-only branches such as PHD rescue, active rescue, shape, or
    MHD association.
    """

    tracker_overrides = {
        "mot17": {},
        "mot20": {},
        "dance": {},
        "sportsmot": {
            "use_soft_phd_weight": True,
            "phd_weight_maha_gate": 12.0,
            "phd_weight_ambiguity_floor": 0.85,
        },
    }

    general_overrides = {
        "mot17": {},
        "mot20": {},
        "dance": {
            "min_box_area": 100,
            "aspect_ratio_thresh": 1000,
        },
        "sportsmot": {},
    }

    default_post_modes = {
        "mot17": "post_gbi",
        "mot20": "post_gbi",
        "dance": "post",
        "sportsmot": "post",
    }

    @classmethod
    def apply_general(cls, dataset: str):
        dataset = canonical_dataset_name(dataset)
        overrides = cls.general_overrides.get(dataset, {})
        if overrides:
            GeneralSettings.dataset_specific_settings.setdefault(dataset, {}).update(deepcopy(overrides))

    @classmethod
    def tracker_cfg(cls, dataset: str):
        dataset = canonical_dataset_name(dataset)
        return deepcopy(cls.tracker_overrides.get(dataset, {}))

    @classmethod
    def default_post_mode(cls, dataset: str) -> str:
        dataset = canonical_dataset_name(dataset)
        return cls.default_post_modes.get(dataset, "post_gbi")
