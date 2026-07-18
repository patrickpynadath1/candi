import ast
from pathlib import Path


def test_algorithm_modules_do_not_own_sampling_loops():
    algos_dir = Path(__file__).parents[3] / "algos"
    definitions = []
    for path in algos_dir.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "generate_samples":
                    definitions.append(path.name)

    assert definitions == ["trainer_base.py"]


def test_each_concrete_sampler_has_its_own_module():
    samplers_dir = Path(__file__).parents[3] / "samplers"
    excluded = {"base.py", "capabilities.py", "registry.py", "runner.py", "utils.py"}
    implementation_files = {
        path.name
        for path in samplers_dir.glob("*.py")
        if path.name not in excluded and path.name != "__init__.py"
    }

    assert implementation_files == {
        "analytic.py",
        "ancestral.py",
        "ancestral_cached.py",
        "autoregressive.py",
        "autoregressive_cached.py",
        "cadd.py",
        "candi.py",
        "candi_cached.py",
        "continuous_embedding.py",
        "continuous_onehot.py",
        "permuted_autoregressive.py",
        "permuted_autoregressive_uncached.py",
        "semi_autoregressive.py",
        "semicat_dopri5.py",
        "semicat_euler.py",
        "semicat_flow_map.py",
        "flm_euler.py",
        "sfm.py",
    }
