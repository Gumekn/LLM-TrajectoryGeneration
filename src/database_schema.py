"""
database_schema.py
数据库设计与接口定义

本模块定义了危险轨迹生成与评估系统的完整数据库结构，包括：
- 轨迹存储表 (TrajectoryStorage)
- 知识库表 (KnowledgeBase)
- 评估结果表 (EvaluationResults)
- 场景元数据表 (ScenarioMetadata)
- 用户审核记录表 (HumanReviewRecords)

支持的数据库类型：
- SQLite: 本地轻量级存储，适合开发和测试
- PostgreSQL: 生产环境关系型数据库
- ChromaDB: 向量数据库，用于RAG检索
- Milvus: 高性能向量数据库（可选）
"""

import os
import json
import sqlite3
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import pickle


# =================================================================
# 枚举类型定义
# =================================================================
class DangerType(Enum):
    """危险类型枚举"""
    REAR_END = "rear_end"           # 追尾
    CUT_IN = "cut_in"               # 切入
    HEAD_ON = "head_on"             # 对向碰撞
    SIDE_SWIPE = "side_swipe"       # 侧面剐蹭
    PEDESTRIAN = "pedestrian"       # 行人碰撞
    CYCLIST = "cyclist"             # 骑车人碰撞
    UNKNOWN = "unknown"


class LabelStatus(Enum):
    """标签状态枚举"""
    REASONABLE = "reasonable"       # 合理
    UNREASONABLE = "unreasonable"   # 不合理
    PENDING = "pending"             # 待审核
    MODIFIED = "modified"           # 已修改


class EvaluatedBy(Enum):
    """评估来源枚举"""
    HUMAN = "human"                 # 人工评估
    LLM_AUTO = "llm_auto"           # LLM自动评估
    RAG = "rag"                     # RAG评估


class ReviewStatus(Enum):
    """审核状态枚举"""
    APPROVED = "approved"           # 已通过
    REJECTED = "rejected"           # 已拒绝
    PENDING = "pending"             # 待审核
    SKIPPED = "skipped"             # 已跳过


# =================================================================
# 数据类定义
# =================================================================
@dataclass
class TrajectoryPoint:
    """轨迹点数据结构"""
    timestamp: float
    global_center: List[float]          # [x, y, z]
    heading: float
    local_velocity: List[float]         # [vx, vy]
    local_acceleration: List[float]     # [ax, ay]
    size: List[float]                   # [length, width, height]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrajectoryPoint':
        return cls(**data)


@dataclass
class TrajectoryFeatures:
    """12维轨迹特征向量"""
    mean_speed: float
    max_speed: float
    speed_std: float
    mean_accel: float
    max_accel: float
    max_lateral_offset: float
    lateral_std: float
    min_ttc: float
    mean_ttc: float
    trajectory_length: float
    max_curvature: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.mean_speed, self.max_speed, self.speed_std,
            self.mean_accel, self.max_accel,
            self.max_lateral_offset, self.lateral_std,
            self.min_ttc, self.mean_ttc,
            self.trajectory_length, self.max_curvature
        ])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'TrajectoryFeatures':
        return cls(*vector.tolist())


@dataclass
class TrajectoryRecord:
    """轨迹记录数据类"""
    # 主键
    trajectory_id: str

    # 外键关联
    scenario_id: str
    vehicle_id: str
    parent_trajectory_id: Optional[str] = None  # 父轨迹ID（用于变异追踪）

    # 轨迹数据
    trajectory_points: List[TrajectoryPoint] = field(default_factory=list)
    feature_vector: Optional[np.ndarray] = None

    # 元数据
    danger_type: DangerType = DangerType.UNKNOWN
    danger_level: str = "low"  # low/medium/high
    generation_batch: str = "default"

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def compute_hash(self) -> str:
        """计算轨迹哈希值，用于去重"""
        data = json.dumps([p.to_dict() for p in self.trajectory_points], sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()


@dataclass
class KnowledgeBaseRecord:
    """知识库记录数据类"""
    # 主键
    case_id: str

    # 外键关联
    trajectory_id: str

    # 向量化特征（用于RAG检索）
    embedding: np.ndarray

    # 标签与评估
    label: LabelStatus
    danger_type: DangerType
    danger_level: str = "low"  # high/medium/low
    evaluated_by: EvaluatedBy
    confidence: float

    # 评估理由
    evaluation_reasoning: str = ""

    # 生成参数上下文（用于可解释性）
    creation_context: str = ""  # JSON格式存储生成参数

    # 相似案例引用
    similar_cases: List[str] = field(default_factory=list)

    # 统计信息
    query_count: int = 0  # 被查询次数

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResultRecord:
    """评估结果记录数据类"""
    # 主键
    evaluation_id: str

    # 外键关联
    trajectory_id: str

    # 物理验证结果
    physics_validation: Dict = field(default_factory=dict)
    is_physics_valid: bool = False

    # RAG评估结果
    max_similarity: float = 0.0
    avg_similarity: float = 0.0
    similar_cases: List[Dict] = field(default_factory=list)

    # LLM评估结果
    llm_is_reasonable: bool = False
    llm_confidence: float = 0.0
    llm_reasoning: str = ""

    # 综合评估
    needs_human_review: bool = True
    final_status: ReviewStatus = ReviewStatus.PENDING

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScenarioMetadataRecord:
    """场景元数据记录数据类"""
    # 主键
    scenario_id: str

    # 场景基本信息
    source_dataset: str = "waymo"  # waymo/nuscenes/etc.
    map_name: str = ""
    scene_length: int = 0  # 帧数
    sampling_rate: float = 10.0  # Hz

    # 车辆信息
    ego_vehicle_id: str = ""
    num_vehicles: int = 0
    num_pedestrians: int = 0
    num_cyclists: int = 0

    # 处理状态
    is_processed: bool = False
    processing_stage: str = "pending"  # pending/processing/completed/failed

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None


@dataclass
class HumanReviewRecord:
    """人工审核记录数据类"""
    # 主键
    review_id: str

    # 外键关联
    trajectory_id: str
    evaluation_id: str

    # 审核信息
    reviewer_id: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_notes: str = ""

    # 修正后的标签（如果需要）
    corrected_label: Optional[LabelStatus] = None

    # 时间戳
    reviewed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


# =================================================================
# 数据库管理器抽象基类
# =================================================================
class DatabaseManager(ABC):
    """数据库管理器抽象基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接数据库"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开数据库连接"""
        pass

    @abstractmethod
    def create_tables(self):
        """创建所有表结构"""
        pass


# =================================================================
# SQLite数据库管理器
# =================================================================
class SQLiteManager(DatabaseManager):
    """
    SQLite数据库管理器
    用于本地轻量级存储，适合开发和测试环境
    """

    def __init__(self, db_path: str = "./data/trajectory_database.db"):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def connect(self) -> bool:
        """连接SQLite数据库"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            print(f"[SQLite] 已连接到数据库: {self.db_path}")
            return True
        except Exception as e:
            print(f"[SQLite] 连接失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
            print("[SQLite] 已断开连接")

    def create_tables(self):
        """创建所有表结构"""

        # 1. 轨迹存储表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectory_records (
                trajectory_id TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                parent_trajectory_id TEXT,
                trajectory_data BLOB,  -- 序列化的轨迹点列表
                feature_vector BLOB,   -- 12维特征向量
                danger_type TEXT,
                danger_level TEXT,
                generation_batch TEXT,
                trajectory_hash TEXT,  -- 用于去重
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_trajectory_id) REFERENCES trajectory_records(trajectory_id)
            )
        """)

        # 2. 知识库表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                case_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                embedding BLOB,         -- 序列化的向量
                label TEXT NOT NULL,
                danger_type TEXT,
                danger_level TEXT DEFAULT 'low',
                evaluated_by TEXT,
                confidence REAL,
                evaluation_reasoning TEXT,
                creation_context TEXT,   -- JSON格式存储生成参数
                similar_cases TEXT,     -- JSON格式存储相似案例ID列表
                query_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trajectory_id) REFERENCES trajectory_records(trajectory_id)
            )
        """)

        # 3. 评估结果表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                evaluation_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                physics_validation TEXT,  -- JSON格式
                is_physics_valid BOOLEAN,
                max_similarity REAL,
                avg_similarity REAL,
                similar_cases TEXT,       -- JSON格式
                llm_is_reasonable BOOLEAN,
                llm_confidence REAL,
                llm_reasoning TEXT,
                needs_human_review BOOLEAN DEFAULT 1,
                final_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trajectory_id) REFERENCES trajectory_records(trajectory_id)
            )
        """)

        # 4. 场景元数据表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenario_metadata (
                scenario_id TEXT PRIMARY KEY,
                source_dataset TEXT,
                map_name TEXT,
                scene_length INTEGER,
                sampling_rate REAL,
                ego_vehicle_id TEXT,
                num_vehicles INTEGER,
                num_pedestrians INTEGER,
                num_cyclists INTEGER,
                is_processed BOOLEAN DEFAULT 0,
                processing_stage TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
        """)

        # 5. 人工审核记录表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_reviews (
                review_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,
                reviewer_id TEXT,
                review_status TEXT,
                reviewer_notes TEXT,
                corrected_label TEXT,
                reviewed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trajectory_id) REFERENCES trajectory_records(trajectory_id),
                FOREIGN KEY (evaluation_id) REFERENCES evaluation_results(evaluation_id)
            )
        """)

        # 创建索引以提高查询效率
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory_scenario
            ON trajectory_records(scenario_id)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory_danger_type
            ON trajectory_records(danger_type)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_label
            ON knowledge_base(label)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluation_status
            ON evaluation_results(final_status)
        """)

        self.connection.commit()
        print("[SQLite] 所有表创建完成")

    # ==================== 轨迹记录操作 ====================

    def insert_trajectory(self, record: TrajectoryRecord) -> bool:
        """插入轨迹记录"""
        try:
            trajectory_data = pickle.dumps(record.trajectory_points)
            feature_vector = pickle.dumps(record.feature_vector) if record.feature_vector is not None else None

            self.cursor.execute("""
                INSERT OR REPLACE INTO trajectory_records
                (trajectory_id, scenario_id, vehicle_id, parent_trajectory_id,
                 trajectory_data, feature_vector, danger_type, danger_level,
                 generation_batch, trajectory_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.trajectory_id,
                record.scenario_id,
                record.vehicle_id,
                record.parent_trajectory_id,
                trajectory_data,
                feature_vector,
                record.danger_type.value,
                record.danger_level,
                record.generation_batch,
                record.compute_hash(),
                record.created_at,
                record.updated_at
            ))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"[SQLite] 插入轨迹记录失败: {e}")
            return False

    def get_trajectory(self, trajectory_id: str) -> Optional[TrajectoryRecord]:
        """获取轨迹记录"""
        try:
            self.cursor.execute("""
                SELECT * FROM trajectory_records WHERE trajectory_id = ?
            """, (trajectory_id,))

            row = self.cursor.fetchone()
            if row:
                return self._row_to_trajectory_record(row)
            return None
        except Exception as e:
            print(f"[SQLite] 获取轨迹记录失败: {e}")
            return None

    def get_trajectories_by_scenario(self, scenario_id: str) -> List[TrajectoryRecord]:
        """获取场景下的所有轨迹"""
        try:
            self.cursor.execute("""
                SELECT * FROM trajectory_records WHERE scenario_id = ?
            """, (scenario_id,))

            rows = self.cursor.fetchall()
            return [self._row_to_trajectory_record(row) for row in rows]
        except Exception as e:
            print(f"[SQLite] 获取场景轨迹失败: {e}")
            return []

    def _row_to_trajectory_record(self, row: sqlite3.Row) -> TrajectoryRecord:
        """将数据库行转换为TrajectoryRecord"""
        return TrajectoryRecord(
            trajectory_id=row['trajectory_id'],
            scenario_id=row['scenario_id'],
            vehicle_id=row['vehicle_id'],
            parent_trajectory_id=row['parent_trajectory_id'],
            trajectory_points=pickle.loads(row['trajectory_data']),
            feature_vector=pickle.loads(row['feature_vector']) if row['feature_vector'] else None,
            danger_type=DangerType(row['danger_type']),
            danger_level=row['danger_level'],
            generation_batch=row['generation_batch'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )

    # ==================== 知识库操作 ====================

    def insert_knowledge_case(self, record: KnowledgeBaseRecord) -> bool:
        """插入知识库案例"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO knowledge_base
                (case_id, trajectory_id, embedding, label, danger_type, danger_level,
                 evaluated_by, confidence, evaluation_reasoning, creation_context, similar_cases,
                 query_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.case_id,
                record.trajectory_id,
                pickle.dumps(record.embedding),
                record.label.value,
                record.danger_type.value,
                record.danger_level,
                record.evaluated_by.value,
                record.confidence,
                record.evaluation_reasoning,
                record.creation_context,
                json.dumps(record.similar_cases),
                record.query_count,
                record.created_at,
                record.updated_at
            ))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"[SQLite] 插入知识库案例失败: {e}")
            return False

    def get_knowledge_case(self, case_id: str) -> Optional[KnowledgeBaseRecord]:
        """获取知识库案例"""
        try:
            self.cursor.execute("""
                SELECT * FROM knowledge_base WHERE case_id = ?
            """, (case_id,))

            row = self.cursor.fetchone()
            if row:
                return self._row_to_knowledge_record(row)
            return None
        except Exception as e:
            print(f"[SQLite] 获取知识库案例失败: {e}")
            return None

    def get_all_knowledge_cases(self, label: LabelStatus = None) -> List[KnowledgeBaseRecord]:
        """获取所有知识库案例（可按标签筛选）"""
        try:
            if label:
                self.cursor.execute("""
                    SELECT * FROM knowledge_base WHERE label = ?
                """, (label.value,))
            else:
                self.cursor.execute("SELECT * FROM knowledge_base")

            rows = self.cursor.fetchall()
            return [self._row_to_knowledge_record(row) for row in rows]
        except Exception as e:
            print(f"[SQLite] 获取知识库案例失败: {e}")
            return []

    def increment_query_count(self, case_id: str):
        """增加案例查询计数"""
        try:
            self.cursor.execute("""
                UPDATE knowledge_base
                SET query_count = query_count + 1
                WHERE case_id = ?
            """, (case_id,))
            self.connection.commit()
        except Exception as e:
            print(f"[SQLite] 更新查询计数失败: {e}")

    def _row_to_knowledge_record(self, row: sqlite3.Row) -> KnowledgeBaseRecord:
        """将数据库行转换为KnowledgeBaseRecord"""
        return KnowledgeBaseRecord(
            case_id=row['case_id'],
            trajectory_id=row['trajectory_id'],
            embedding=pickle.loads(row['embedding']),
            label=LabelStatus(row['label']),
            danger_type=DangerType(row['danger_type']),
            danger_level=row.get('danger_level', 'low'),
            evaluated_by=EvaluatedBy(row['evaluated_by']),
            confidence=row['confidence'],
            evaluation_reasoning=row['evaluation_reasoning'],
            creation_context=row.get('creation_context', ''),
            similar_cases=json.loads(row['similar_cases']) if row['similar_cases'] else [],
            query_count=row['query_count'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )

    # ==================== 评估结果操作 ====================

    def insert_evaluation_result(self, record: EvaluationResultRecord) -> bool:
        """插入评估结果"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO evaluation_results
                (evaluation_id, trajectory_id, physics_validation, is_physics_valid,
                 max_similarity, avg_similarity, similar_cases, llm_is_reasonable,
                 llm_confidence, llm_reasoning, needs_human_review, final_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.evaluation_id,
                record.trajectory_id,
                json.dumps(record.physics_validation),
                record.is_physics_valid,
                record.max_similarity,
                record.avg_similarity,
                json.dumps(record.similar_cases),
                record.llm_is_reasonable,
                record.llm_confidence,
                record.llm_reasoning,
                record.needs_human_review,
                record.final_status.value,
                record.created_at
            ))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"[SQLite] 插入评估结果失败: {e}")
            return False

    def get_evaluation_result(self, evaluation_id: str) -> Optional[EvaluationResultRecord]:
        """获取评估结果"""
        try:
            self.cursor.execute("""
                SELECT * FROM evaluation_results WHERE evaluation_id = ?
            """, (evaluation_id,))

            row = self.cursor.fetchone()
            if row:
                return self._row_to_evaluation_record(row)
            return None
        except Exception as e:
            print(f"[SQLite] 获取评估结果失败: {e}")
            return None

    def _row_to_evaluation_record(self, row: sqlite3.Row) -> EvaluationResultRecord:
        """将数据库行转换为EvaluationResultRecord"""
        return EvaluationResultRecord(
            evaluation_id=row['evaluation_id'],
            trajectory_id=row['trajectory_id'],
            physics_validation=json.loads(row['physics_validation']) if row['physics_validation'] else {},
            is_physics_valid=bool(row['is_physics_valid']),
            max_similarity=row['max_similarity'],
            avg_similarity=row['avg_similarity'],
            similar_cases=json.loads(row['similar_cases']) if row['similar_cases'] else [],
            llm_is_reasonable=bool(row['llm_is_reasonable']),
            llm_confidence=row['llm_confidence'],
            llm_reasoning=row['llm_reasoning'],
            needs_human_review=bool(row['needs_human_review']),
            final_status=ReviewStatus(row['final_status']),
            created_at=datetime.fromisoformat(row['created_at'])
        )

    # ==================== 场景元数据操作 ====================

    def insert_scenario_metadata(self, record: ScenarioMetadataRecord) -> bool:
        """插入场景元数据"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO scenario_metadata
                (scenario_id, source_dataset, map_name, scene_length, sampling_rate,
                 ego_vehicle_id, num_vehicles, num_pedestrians, num_cyclists,
                 is_processed, processing_stage, created_at, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.scenario_id,
                record.source_dataset,
                record.map_name,
                record.scene_length,
                record.sampling_rate,
                record.ego_vehicle_id,
                record.num_vehicles,
                record.num_pedestrians,
                record.num_cyclists,
                record.is_processed,
                record.processing_stage,
                record.created_at,
                record.processed_at
            ))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"[SQLite] 插入场景元数据失败: {e}")
            return False

    def update_scenario_status(self, scenario_id: str, stage: str, is_processed: bool = False):
        """更新场景处理状态"""
        try:
            processed_at = datetime.now() if is_processed else None
            self.cursor.execute("""
                UPDATE scenario_metadata
                SET processing_stage = ?, is_processed = ?, processed_at = ?
                WHERE scenario_id = ?
            """, (stage, is_processed, processed_at, scenario_id))
            self.connection.commit()
        except Exception as e:
            print(f"[SQLite] 更新场景状态失败: {e}")

    # ==================== 人工审核记录操作 ====================

    def insert_human_review(self, record: HumanReviewRecord) -> bool:
        """插入人工审核记录"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO human_reviews
                (review_id, trajectory_id, evaluation_id, reviewer_id, review_status,
                 reviewer_notes, corrected_label, reviewed_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.review_id,
                record.trajectory_id,
                record.evaluation_id,
                record.reviewer_id,
                record.review_status.value,
                record.reviewer_notes,
                record.corrected_label.value if record.corrected_label else None,
                record.reviewed_at,
                record.created_at
            ))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"[SQLite] 插入审核记录失败: {e}")
            return False


# =================================================================
# 向量数据库接口（预留）
# =================================================================
class VectorDatabaseInterface(ABC):
    """
    向量数据库接口抽象基类
    用于RAG检索，支持ChromaDB、Milvus等
    """

    @abstractmethod
    def connect(self) -> bool:
        """连接向量数据库"""
        pass

    @abstractmethod
    def add_vectors(self, ids: List[str], vectors: List[np.ndarray], metadatas: List[Dict]):
        """添加向量"""
        pass

    @abstractmethod
    def query_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        查询相似向量

        Returns:
            List of (id, similarity, metadata) tuples
        """
        pass


class ChromaDBInterface(VectorDatabaseInterface):
    """
    ChromaDB接口实现（预留）
    需要安装: pip install chromadb
    """

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "trajectory_cases"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def connect(self) -> bool:
        """连接ChromaDB"""
        try:
            # 预留：实际使用时取消注释
            # import chromadb
            # from chromadb.config import Settings
            #
            # self.client = chromadb.Client(
            #     Settings(persist_directory=self.persist_directory)
            # )
            # self.collection = self.client.get_or_create_collection(
            #     name=self.collection_name
            # )
            print(f"[ChromaDB] 已连接到数据库: {self.persist_directory}")
            return True
        except Exception as e:
            print(f"[ChromaDB] 连接失败: {e}")
            return False

    def add_vectors(self, ids: List[str], vectors: List[np.ndarray], metadatas: List[Dict]):
        """添加向量到ChromaDB"""
        # 预留：实际使用时取消注释
        # if self.collection:
        #     self.collection.add(
        #         ids=ids,
        #         embeddings=[v.tolist() for v in vectors],
        #         metadatas=metadatas
        #     )
        pass

    def query_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """查询相似向量"""
        # 预留：实际使用时取消注释
        # if self.collection:
        #     results = self.collection.query(
        #         query_embeddings=[query_vector.tolist()],
        #         n_results=top_k
        #     )
        #     return [
        #         (id, 1 - dist, meta)
        #         for id, dist, meta in zip(
        #             results['ids'][0],
        #             results['distances'][0],
        #             results['metadatas'][0]
        #         )
        #     ]
        return []


# =================================================================
# 数据库连接工厂
# =================================================================
class DatabaseFactory:
    """数据库连接工厂"""

    @staticmethod
    def create_sqlite_manager(db_path: str = "./data/trajectory_database.db") -> SQLiteManager:
        """创建SQLite管理器"""
        return SQLiteManager(db_path)

    @staticmethod
    def create_vector_db_interface(
        db_type: str = "chromadb",
        **kwargs
    ) -> VectorDatabaseInterface:
        """创建向量数据库接口"""
        if db_type == "chromadb":
            return ChromaDBInterface(**kwargs)
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")


# =================================================================
# 与主程序衔接的便捷函数
# =================================================================
def init_database(db_path: str = "./data/trajectory_database.db") -> SQLiteManager:
    """
    初始化数据库

    Returns:
        SQLiteManager: 已连接并初始化表的数据库管理器
    """
    manager = DatabaseFactory.create_sqlite_manager(db_path)

    if manager.connect():
        manager.create_tables()
        return manager
    else:
        raise ConnectionError("无法连接到数据库")


def save_trajectory_with_evaluation(
    manager: SQLiteManager,
    trajectory_record: TrajectoryRecord,
    evaluation_record: EvaluationResultRecord,
    to_knowledge_base: bool = False
):
    """
    保存轨迹和评估结果（便捷函数）

    Args:
        manager: 数据库管理器
        trajectory_record: 轨迹记录
        evaluation_record: 评估结果记录
        to_knowledge_base: 是否同时添加到知识库
    """
    # 保存轨迹
    manager.insert_trajectory(trajectory_record)

    # 保存评估结果
    manager.insert_evaluation_result(evaluation_record)

    # 如果需要，添加到知识库
    if to_knowledge_base and evaluation_record.final_status == ReviewStatus.APPROVED:
        knowledge_record = KnowledgeBaseRecord(
            case_id=f"case_{trajectory_record.trajectory_id}",
            trajectory_id=trajectory_record.trajectory_id,
            embedding=trajectory_record.feature_vector,
            label=LabelStatus.REASONABLE if evaluation_record.llm_is_reasonable else LabelStatus.UNREASONABLE,
            danger_type=trajectory_record.danger_type,
            evaluated_by=EvaluatedBy.LLM_AUTO,
            confidence=evaluation_record.llm_confidence,
            evaluation_reasoning=evaluation_record.llm_reasoning
        )
        manager.insert_knowledge_case(knowledge_record)


# =================================================================
# 测试代码
# =================================================================
if __name__ == "__main__":
    # 初始化数据库
    db_manager = init_database("./test_database.db")

    # 创建测试数据
    test_trajectory = TrajectoryRecord(
        trajectory_id="test_001",
        scenario_id="scenario_001",
        vehicle_id="vehicle_001",
        trajectory_points=[
            TrajectoryPoint(
                timestamp=0.0,
                global_center=[0.0, 0.0, 0.0],
                heading=0.0,
                local_velocity=[10.0, 0.0],
                local_acceleration=[0.0, 0.0],
                size=[4.5, 2.0, 1.5]
            )
        ],
        feature_vector=np.array([1.0] * 12),
        danger_type=DangerType.REAR_END,
        danger_level="high"
    )

    # 插入测试数据
    success = db_manager.insert_trajectory(test_trajectory)
    print(f"插入测试结果: {success}")

    # 查询测试数据
    retrieved = db_manager.get_trajectory("test_001")
    print(f"查询结果: {retrieved}")

    # 关闭连接
    db_manager.disconnect()
