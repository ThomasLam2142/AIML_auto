from transformers import BertTokenizer, FlaxBertForMultipleChoice
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

