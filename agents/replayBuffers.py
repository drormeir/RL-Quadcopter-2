import random
from collections import namedtuple, deque
random.seed(1)

# works for any memory size and any batch size
def replay_memory_sample(memory, batch_size):
    len_mem = len(memory)
    if (len_mem < 1) or (batch_size < 1):
        return []
    sub_batch_size = batch_size % len_mem
    mem_times      = batch_size // len_mem
    return list(memory) * mem_times + random.sample(memory, k=sub_batch_size)


class GoodBadReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        half_buffer_size      = (int(buffer_size) + 1) // 2
        self.half_batch_size  = (int(batch_size) + 1)  // 2
        self.full_batch_size  = self.half_batch_size * 2
        self.good_memory      = deque(maxlen=half_buffer_size)  # internal memory (deque)
        self.bad_memory       = deque(maxlen=half_buffer_size)  # internal memory (deque)
        self.experience       = namedtuple("Experience", field_names=["state", "action", "diff_reward",\
                                                                      "step_reward", "next_state", "done"])

    def add(self, state, action, diff_reward, step_reward, next_state, done):
        """Add a new experience to memory."""
        if (abs(diff_reward) < 1e-8):
            return
        e = self.experience(state, action, diff_reward, step_reward, next_state, done)
        if diff_reward < 0:
            self.bad_memory.append(e)
        else:
            self.good_memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        len_good   = len(self.good_memory)
        len_bad    = len(self.bad_memory)
        if (len_good < 1):
            return replay_memory_sample(self.bad_memory, self.full_batch_size)
        if (len_bad < 1):
            return replay_memory_sample(self.good_memory, self.full_batch_size)
        return replay_memory_sample(self.good_memory, self.half_batch_size) + replay_memory_sample(self.bad_memory,\
                                                                                                   self.half_batch_size)

    def has_sample(self):
        return len(self.good_memory) + len(self.bad_memory) >= self.full_batch_size
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.good_memory) + len(self.bad_memory)

# original source code

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory     = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

