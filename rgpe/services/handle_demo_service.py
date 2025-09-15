from ..demos.svr_v1_demo import SVRV1Demo
from ..demos.svr_v2_demo import SVRV2Demo
from ..demos.svr_v3_demo import SVRV3Demo
from ..demos.qsvm_pennylane_demo import QSVMPennylaneDemo
from ..demos.qsvr_qiskit_demo import QSVRQiskitDemo
from ..demos.base_demo import BaseDemo


class HandleDemoService:
    demos: dict[str, type[BaseDemo]] = {
        "svr_v1": SVRV1Demo,
        "svr_v2": SVRV2Demo,
        "svr_v3": SVRV3Demo,
        "qsvm_pennylane": QSVMPennylaneDemo,
        "qsvr_qiskit": QSVRQiskitDemo
    }

    @classmethod
    def list_demos(cls) -> list[str]:
        return list(cls.demos.keys())

    @classmethod
    def run(cls, demo_key: str, *args, **kwargs) -> None:
        if demo_key not in cls.demos:
            demos_keys = cls.list_demos()
            error_text = f"Demo '{demo_key}' não encontrada. Demos disponíveis: {demos_keys}"
            raise ValueError(error_text)

        demo = cls.demos[demo_key](*args, **kwargs)
        demo.run()
