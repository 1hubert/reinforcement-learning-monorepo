import logging

def test():
    logging.debug('debug info')
    logging.info('info')
    logging.error('something went wrong i guess')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(asctime)s - %(message)s'
    )
    test()
