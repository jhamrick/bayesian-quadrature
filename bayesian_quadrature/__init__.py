import logging
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("INFO")


from bq import BQ
__all__ = ['BQ']
