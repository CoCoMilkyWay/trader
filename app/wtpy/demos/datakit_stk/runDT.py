import time
from wtpy import WtDtEngine

if __name__ == "__main__":
    #创建一个运行环境，并加入策略
    env = WtDtEngine()
    env.initialize()
    
    env.run(True)

    print('press ctrl-c to exit')
    try:
    	while True:
            time.sleep(1)
    except KeyboardInterrupt as e:
    	exit(0)