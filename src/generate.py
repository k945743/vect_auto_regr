import argparse
from autoreg import VectAutoReg

if __name__ == "__main__":
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_count", type=str, required=True)
    parser.add_argument("--sample_length", type=str, required=True)

    args = parser.parse_args()
