import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='pairwiseAircraft-v0',
    entry_point='gym_pairwiseAircraft.envs:pairwiseAircraftEnv'
)
