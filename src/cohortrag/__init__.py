# -*- coding: utf-8 -*-
"""COHORT-RAG：面向条件聚合问答的可复用证据执行器（CPU 核心实现）。

对应草案《COHORT-RAG》第 5—11 节的执行核心。所有模块均为真实、可运行的实现；
LLM 语义算子为可插拔接口（semantics.py），接入本地模型即可在 GPU 机器上运行。
"""
from .snapshot import Snapshot
from .ledger import EvidenceLedger
from .cs import FinitePopulationCS
from .projection import (build_polytope, min_max_B, ratio_bounds,
                         threshold_margins, verify_outward_enclosure, FeasibilityError)
from .budget import (BudgetContext, QueryBudgetState, RoundRobinAllocator,
                     WorstVarianceAllocator, CertificateGuidedAllocator,
                     SmallStratumFirstAllocator, DrainAwareRoundRobinAllocator,
                     ProportionalAllocator)
from .pipeline import COHORTRAG, Config, QueryResult
from .workload import (Query, make_workload, make_cfpb_workload, make_wands_workload)
from .render import render_claim, template_text, certificate
from . import contracts, datasets
from .llm_operator import (LocalLLMOperator, CachedSemanticOperator,
                           LabelTableOperator, LabelFailure)

__all__ = [
    "Snapshot", "EvidenceLedger", "FinitePopulationCS",
    "build_polytope", "min_max_B", "ratio_bounds", "FeasibilityError",
    "BudgetContext", "QueryBudgetState", "RoundRobinAllocator", "WorstVarianceAllocator",
    "COHORTRAG", "Config", "QueryResult",
    "Query", "make_workload", "make_cfpb_workload", "make_wands_workload",
    "render_claim", "template_text", "certificate",
    "contracts", "datasets",
    "LocalLLMOperator", "CachedSemanticOperator", "LabelTableOperator", "LabelFailure",
]
