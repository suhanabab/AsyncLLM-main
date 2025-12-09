from abc import ABC, abstractmethod
from utils.sys_utils import device_config, comm_config

class BaseClient(ABC):
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.server = None
        self.dataset = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def run(self, model):
        pass

    @abstractmethod
    def local_test(self, model):
        pass


class BaseServer(ABC):
    def __init__(self, args, clients):
        self.args = args
        self.clients = clients

        # === 加入慢速设备配置 ===
        delays = device_config(len(clients))
        bandwidths = comm_config(len(clients))

        print("\n========== Device Configuration ==========")
        for client in self.clients:
            client.delay = delays[client.id]
            client.bandwidth = bandwidths[client.id]
            client.training_time = 0

            dev_type = "SLOW device" if client.delay > 1 else "FAST device"
            print(
                f"Client {client.id}: delay ×{client.delay:.2f}, bandwidth={client.bandwidth:.2f}MB/s  -->  {dev_type}")
        print("==========================================\n")

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def local_run(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def test_all(self):
        pass