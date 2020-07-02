rx = QRegExp("\d{10}")
MGRSvalid = QRegExpValidator(rx, self)


def MGRS2meter(MGRScoord):
    #MRGS 좌표를 원점으로부터의 미터 거리로 환산하는 함수
    #input: 5자리 숫자로 된 string(MGRS 좌표)
    #output: int(미터)
    return int(MGRScood)