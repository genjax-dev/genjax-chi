import jax
import time
import multiprocessing


def i_wont_be_so_fine(x):
    return jax.lax.while_loop(lambda x: x > 0, lambda x: x * x, x)


def run_process():
    p = multiprocessing.Process(target=i_wont_be_so_fine, args=(1.0,))
    p.start()
    time.sleep(10)
    if p.is_alive():
        print("I'm still running")
        p.terminate()
        p.join()


if __name__ == "__main__":
    # Run the process
    run_process()
