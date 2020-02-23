def gen(i):
	print("List for %s values created" % i)
	return [ 0 for i in range ( ( i >> 3) + (0 != (i & 7) ) ) ]
	#return i

def set(l, m):
	print("Set %s exists in map..." % m)
	l[int(m/8)] = l[int(m/8)] | 1 << (m & 7)
	return l


def find(j, k):
	print("Does %s exist in map?: %s" % (k,(( j[int(k/8)] & 1 << (k & 7)) != 0)))
	print(j)


if __name__ == "__main__":
	listg = gen(3)
	listg = set(listg, 3)
	find(listg, 3)
