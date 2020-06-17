import subprocess
import time

if __name__ == "__main__":
    # 0, 11, 32, 100, 103, 254, 273, 301, 568, 1812
    seeds = [0, 11, 32, 100, 103, 254, 273, 301, 568, 1812]
    # 0.002, 0.004, 0.008, 0.016, 0.032, 0.063, 0.126, 0.251, 0.501, 1.000
    percentages = [0.002, 0.004, 0.008, 0.016, 0.032, 0.063, 0.126, 0.251, 0.501, 1.000]

    for percent in percentages:
        for seed in seeds:
            time.sleep(2)
            args = [
                "C:\\Users\\Freddie\\anaconda3\\envs\\tf_gpu\\python",
                "./train.py",
                "-c",
                "conf/training_setup.yaml",
                "--seed",
                "{}".format(seed),
                "--percent-traning-data",
                "{:.3f}".format(percent),
            ]

            print(seed, end=" ")
            print(percent, end=" ")
            try:
                result = subprocess.run(
                    args, shell=True, check=True, capture_output=True,
                )
            except Exception as e:
                print(e)
                continue

            print(result.stdout.splitlines()[-2].decode("UTF8"), end=" ")
            print(result.stdout.splitlines()[-1].decode("UTF8"))
