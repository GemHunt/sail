import lmdb

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


def write_batch_to_lmdb(db, batch):
    """
    Write a batch of (key,value) to a lmdb
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for key, datum in batch:
                lmdb_txn.put(key, datum.SerializeToString())
    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit * 2
        try:
            db.set_mapsize(new_limit)  # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0, 87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        write_batch_to_lmdb(db, batch)
