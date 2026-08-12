import pytest


torch = pytest.importorskip("torch")

from src.logic import SoftThresholdPredicate, TensorFraudKnowledgeBase, TensorLogic


def test_tensor_logic_range_and_gradient():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    predicate = SoftThresholdPredicate(1.5, temperature=0.5, learnable=True)
    truth = predicate(values)
    satisfaction = TensorLogic.forall(truth)
    (1.0 - satisfaction).backward()
    assert ((truth >= 0.0) & (truth <= 1.0)).all()
    assert predicate.threshold.grad is not None
    assert torch.isfinite(predicate.threshold.grad)


def test_tensor_knowledge_base_output():
    knowledge_base = TensorFraudKnowledgeBase(
        predicates={
            "high_amount": SoftThresholdPredicate(1.0, 0.5),
            "high_velocity": SoftThresholdPredicate(2.0, 0.5),
        },
        rules={"compound_risk": ("high_amount", "high_velocity")},
    )
    truth = knowledge_base(
        {
            "high_amount": torch.tensor([0.0, 2.0]),
            "high_velocity": torch.tensor([1.0, 3.0]),
        }
    )
    assert set(truth) == {"compound_risk"}
    assert truth["compound_risk"].shape == (2,)
    assert truth["compound_risk"][1] > truth["compound_risk"][0]
