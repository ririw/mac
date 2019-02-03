from mac import utils, train

_used_no_del_by_flake8_ = [
    train
]


def main() -> None:
    utils.MAC.run()


if __name__ == '__main__':
    main()
