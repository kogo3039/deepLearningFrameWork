# =============================================================================
# step23.py부터 step32.py까지는 simple_core를 이용해야 합니다.
is_simple_core = False  # True
# =============================================================================

if is_simple_core:
    from kogo.core_simple import Variable
    from kogo.core_simple import Function
    from kogo.core_simple import using_config
    from kogo.core_simple import no_grad
    from kogo.core_simple import as_array
    from kogo.core_simple import as_variable
    from kogo.core_simple import setup_variable

else:
    from kogo.core import Variable
    from kogo.core import Parameter
    from kogo.core import Function
    from kogo.core import using_config
    from kogo.core import no_grad
    from kogo.core import test_mode
    from kogo.core import as_array
    from kogo.core import as_variable
    from kogo.core import setup_variable
    from kogo.core import Config
    from kogo.layers import Layer
    from kogo.models import Model
    from kogo.datasets import Dataset
    from kogo.dataloaders import DataLoader
    from kogo.dataloaders import SeqDataLoader

    import kogo.datasets
    import kogo.dataloaders
    import kogo.optimizers
    import kogo.functions
    import kogo.functions_conv
    import kogo.layers
    import kogo.utils
    import kogo.cuda
    import kogo.transforms

setup_variable()
__version__ = '0.0.13'
