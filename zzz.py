def _todir(path):
    if len(path) == 0:
        return path
    if not path[-1] == '/':
        path = path + '/'
    return path
