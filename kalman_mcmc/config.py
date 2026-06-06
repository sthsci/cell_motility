import jax
import matplotlib as mpl
import numpyro

numpyro.set_host_device_count(12)
mpl.use("Agg")

from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams.update({"figure.autolayout": True})

jax.config.update("jax_enable_x64", False)
