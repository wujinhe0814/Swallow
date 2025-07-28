slot_len = 1000
def fun(times, sizes):
    load_time = times[-1]

    packet_count_matrix = [[0 for i in range(slot_len)], [0 for i in range(slot_len)]]  # 每行 [上行包, 下行包]
    slot_duration =  3 * load_time / slot_len
    min_time_slot = 0.02
    max_time_slot = 0.08
    if slot_duration <= min_time_slot:
        slot_duration = min_time_slot
    if slot_duration >= max_time_slot:
        slot_duration = max_time_slot
    for i, t in enumerate(times):
        slot_index = int(t // slot_duration)
        if slot_index >= slot_len:
            break
        if sizes[i] == 1:
            packet_count_matrix[0][slot_index] += 1  # 上行包计数
        else:
            packet_count_matrix[1][slot_index] += 1  # 下行包计数
    return packet_count_matrix,load_time,slot_duration