from dataclasses import dataclass, field
import yaml

@dataclass
class FeatureConfig: 
    query_level: list[str]
    item_level: list[str]
    interaction_level: list[str]

    def all_features(self) -> list[str]:
        return self.query_level + self.item_level + self.interaction_level
    
@dataclass
class DataConfig:
    raw_path: str
    group_col: str
    item_col: str
    position_col: str
    label_cols: list[str]
    feature_cols: FeatureConfig

@dataclass
class SplitConfig:
    val_frac: float
    test_frac: float
    seed: int

@dataclass
class PipelineConfig:
    data: DataConfig
    split: SplitConfig
    output_dir: str

    @staticmethod
    def from_yaml(path: str) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return PipelineConfig(
            data=DataConfig(
                feature_cols=FeatureConfig(**raw["data"]["feature_cols"]),
                **{k: v for k, v in raw["data"].items() if k != "feature_cols"},
            ),
            split=SplitConfig(**raw["split"]),
            output_dir=raw["output_dir"],
        )