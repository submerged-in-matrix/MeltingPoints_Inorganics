from env.modules import *
# Set environment variables for parallel processing
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

random.seed(42)
np.random.seed(42)
rng = np.random.RandomState(42)