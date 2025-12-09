import importlib.util
import sys
import numpy as np
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from utils.options import args_parser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FedSim:
    def __init__(self, args):
        self.args = args
        args.suffix = f'exp/{args.suffix}'
        os.makedirs(f'./{args.suffix}', exist_ok=True)

        output_path = f'{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                      f'{args.cn}c_{args.epoch}E_lr{args.lr}'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.output = open(f'./{output_path}.txt', 'a')

        # === route to algorithm module ===
        alg_module = importlib.import_module(f'alg.{args.alg}')

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.cn), desc="Loading clients...")]
        self.server = alg_module.Server(args, self.clients)

        # ======================================
        # <<< 新增：用于绘图的存储列表
        # ======================================
        self.loss_list = []
        self.time_list = []
        self.start_time = time.time()
        # ======================================

    def simulate(self):
        TEST_GAP = self.args.test_gap
        TOTAL_ROUNDS = getattr(self.args, 'rnd', 1000)
        sim_start_time = time.time()

        try:
            for rnd in tqdm(range(0, self.args.rnd), desc='Communication round', leave=False):
                round_start = time.time()

                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                if (self.args.rnd - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()

                    avg_loss = ret_dict['loss']
                    avg_perplexity = ret_dict['perplexity']
                    print(f"\n[Round {rnd}] Average Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")

                    self.output.write(f'\n[Round {rnd}] Average Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}')
                    self.output.flush()

                    # ======================================
                    # <<< 新增：记录时间和 loss 以便绘图
                    # ======================================
                    elapsed_time = time.time() - self.start_time
                    self.time_list.append(elapsed_time)
                    self.loss_list.append(avg_loss)
                    # ======================================

                # round summary
                msg_round = f"    >>> Round Total Time: {time.time() - round_start:.3f}s\n"
                print(msg_round)
                self.output.write(msg_round)
                self.output.flush()

        finally:
            sim_end_time = time.time()

            msg_final = f"\n========== Simulation Finished ==========\nTotal Running Time: {sim_end_time - sim_start_time:.3f}s\n"
            print(msg_final)
            self.output.write(msg_final)
            self.output.flush()

            # ======================================
            # <<< 新增：绘制并保存 time-loss 图
            # ======================================
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8,5))
            plt.plot(self.time_list, self.loss_list, label=f"{self.args.alg}", linewidth=2)

            plt.xlabel("Time (seconds)")
            plt.ylabel("Average Loss")
            plt.title("Time-Loss Curve")
            plt.legend()
            plt.grid()

            plot_path = f'./{self.args.suffix}/{self.args.alg}_{self.args.dataset}_{self.args.model}_time_loss.png'
            plt.savefig(plot_path, dpi=300)
            print(f"Time-Loss 图已保存到: {plot_path}")
            # ======================================



if __name__ == '__main__':
    FedSim(args=args_parser()).simulate()