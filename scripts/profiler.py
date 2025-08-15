"""Script to repeatedly run hutoken's encode/decode for profiling."""

import argparse
import logging
import pathlib

import hutoken

logging.basicConfig(level=logging.INFO,
                    format=('%(asctime)s - %(name)s - [%(levelname)s] - '
                            '%(message)s'))

logger = logging.getLogger(__name__)


def read_file(path: pathlib.Path) -> str:
    """Reads the content of the file."""
    if not path.is_file():
        raise FileNotFoundError(f'The file "{path}" does not exist')
    with path.open('r', encoding='utf-8') as f:
        return f.read()


def main() -> None:
    """Run the profiling workload."""
    parser = argparse.ArgumentParser(
        description='A script to repeatedly run hutoken\'s encode/decode for'
        'profiling')

    parser.add_argument('--file-path',
                        type=pathlib.Path,
                        required=True,
                        help='path to the text file for encoding')
    parser.add_argument('--iter',
                        type=int,
                        default=5000,
                        help='number of iterations to run')

    parser.add_argument('--decode',
                        action='store_true',
                        help='run decoding too')

    args = parser.parse_args()

    logger.info('Loading document from: %s.', args.file_path)
    document = read_file(args.file_path)

    logger.debug('Initializing hutoken...')
    hutoken.initialize(
        './vocabs/gpt2-vocab.txt',
        './vocabs/gpt2-vocab_special_chars.txt',
        is_byte_encoder=True,
    )
    logger.debug('Initialization complete.')

    logger.info('Starting profiling loop for %d iterations...', args.iter)
    for _ in range(args.iter):
        encoded_tokens = hutoken.encode(document)

        if args.decode:
            hutoken.decode(encoded_tokens)
    logger.info('Profiling loop finished.')


if __name__ == '__main__':
    main()
