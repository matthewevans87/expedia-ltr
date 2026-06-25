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
    
@dataclass
class LGBMHyperparams:
    objective: str
    metric: str
    ndcg_eval_at: list[int]
    num_leaves: int
    learning_rate: float
    n_estimators: int
    early_stopping_rounds: int
    label_gain: list[float]
    seed: int

@dataclass
class EvalConfig:
    k_values: list[int]

@dataclass
class LGBMRunConfig:
    data: dict
    lgbm: LGBMHyperparams
    eval: EvalConfig
    output_dir: str

    @staticmethod
    def from_yaml(path: str) -> "LGBMRunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return LGBMRunConfig(
            data=raw["data"],
            lgbm=LGBMHyperparams(**raw["lgbm"]),
            eval=EvalConfig(**raw["eval"]),
            output_dir=raw["output_dir"]
        )
    
@dataclass
class TowerFeatureConfig: 
    continuous: list[str]
    categorical: dict

@dataclass
class TwoTowerModelConfig:
    tower_dims: list[int]
    embedding_dim: int
    dropout: float

@dataclass
class TwoTowerTrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int

@dataclass
class TwoTowerRunConfig:
    data: dict
    model: TwoTowerModelConfig
    training: TwoTowerTrainingConfig
    eval: EvalConfig
    output_dir: str

    @staticmethod
    def from_yaml(path:str) -> "TwoTowerRunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return TwoTowerRunConfig(
            data = raw["data"],
            model = TwoTowerModelConfig(**raw["model"]),
            training = TwoTowerTrainingConfig(**raw["training"]),
            eval = EvalConfig(**raw["eval"]),
            output_dir = raw["output_dir"]
        )