def unpack_tuple(tup):
    returned = ""
    for item in tup:
        returned += "," + str(item.item())
    return returned[1:]