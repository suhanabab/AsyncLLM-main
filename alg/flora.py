from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record



class Client(FTBaseClient):

    def __init__(self, id, args):
        super().__init__(id, args)

        if hasattr(args, "local_rank"):
            self.lora_config.r = args.local_rank   
        elif hasattr(args, f"client{id}_rank"):
            self.lora_config.r = getattr(args, f"client{id}_rank")  

    @time_record
    def run(self, model):
        
        super().run(model)


class Server(FTBaseServer):
    pass
