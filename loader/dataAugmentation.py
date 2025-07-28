import random
import numpy as np

def median_of_non_zero(values):
    """
    Calculate the median of non-zero values in a list.
    
    Args:
        values: List of numerical values
        
    Returns:
        float or None: Median of non-zero values, or None if no non-zero values exist
    """
    # Filter out non-zero elements
    non_zero_values = [x for x in values if x != 0]

    # Return None if no non-zero elements exist
    if not non_zero_values:
        return None

    # Sort non-zero elements
    non_zero_values.sort()

    # Calculate median
    n = len(non_zero_values)
    if n % 2 == 1:
        # For odd number of elements, take the middle element
        median = non_zero_values[n // 2]
    else:
        # For even number of elements, take average of middle two elements
        median = (non_zero_values[n // 2 - 1] + non_zero_values[n // 2]) / 2

    return median

class Augmentor:
    """
    Data augmentation class for network traffic analysis.
    Implements three augmentation techniques:
    1. Trace Fluctuation: Modifies packet sizes with random fluctuations
    2. Trace Flatten: Inserts time slots to flatten the traffic pattern
    3. Trace Aggregation: Removes time slots to aggregate traffic patterns
    """
    def __init__(self):
        """Initialize augmentation parameters."""
        # Handshake packet parameters
        self.handshake_packet_sum = 20
        self.remain_zero_prob = 0.5
        self.alpha_upload = 0.7
        self.alpha_download = 0.7
        self.num_window_avg = 20

        # Shrink parameters
        self.shrink_ratio_max = 0.7
        self.shrink_ratio_min = 0.3
        self.shrink_prob = 0.6
        self.insert_prob = 0.1

        # Remove parameters
        self.remove_prob = 0.7
        self.increase_ratio_max = 0.7
        self.increase_ratio_min = 0.3
        self.increase_prob = 0.6

    def trace_fluctuation(self, CIF, load_time, slot_duration):
        """
        Trace Fluctuation: Modifies packet sizes with random fluctuations.
        
        Args:
            CIF: Consistent Interaction Feature matrix
            load_time: Total load time
            slot_duration: Duration of each time slot
            
        Returns:
            numpy.ndarray: Modified traffic data with fluctuated packet sizes
        """
        packet_sum = 0
        beta = 1  # Website sensitivity
        upload_value = CIF[0]
        download_value = CIF[1]

        upload_value_ori = upload_value.copy()
        download_value_ori = download_value.copy()

        for i in range(len(upload_value)):
            # Skip if current time slot exceeds load time
            if i * slot_duration >= load_time:
                break

            time_slot_upload, time_slot_download = upload_value[i], download_value[i]
            packet_sum += time_slot_upload + time_slot_download

            # Skip modification if handshake packets not exceeded
            if packet_sum <= self.handshake_packet_sum:
                continue

            # Process upload packets
            if time_slot_upload == 0:
                if random.random() > self.remain_zero_prob:
                    start = max(0, i - self.num_window_avg)
                    end = min(len(upload_value), i + self.num_window_avg + 1)
                    time_slot_upload = int(np.mean(upload_value[start:end])) if end - start >= self.num_window_avg else 0
            else:
                if random.random() <= 0.5:
                    time_slot_upload = int((1 - self.alpha_upload * random.random()) * time_slot_upload)
                else:
                    time_slot_upload = int((1 + self.alpha_upload * random.random()) * time_slot_upload)

            # Process download packets
            if time_slot_download == 0:
                if random.random() > self.remain_zero_prob:
                    start = max(0, i - self.num_window_avg)
                    end = min(len(download_value), i + self.num_window_avg + 1)
                    time_slot_download = int(np.mean(download_value[start:end])) if end - start >= self.num_window_avg else 0
            else:
                if random.random() <= 0.5:
                    time_slot_download = int((1 - self.alpha_download * beta * random.random()) * time_slot_download)
                else:
                    time_slot_download = int((1 + self.alpha_download * beta * random.random()) * time_slot_download)

            # Update values
            upload_value_ori[i] = time_slot_upload
            download_value_ori[i] = time_slot_download

        return np.vstack([np.array(upload_value_ori), np.array(download_value_ori)])

    def trace_aggregation(self, CIF, load_time, slot_duration):
        """
        Trace Aggregation: Inserts new time slots to aggregate traffic patterns.
        
        Args:
            CIF: Consistent Interaction Feature matrix
            load_time: Total load time
            slot_duration: Duration of each time slot
            
        Returns:
            numpy.ndarray: Modified traffic data with aggregated patterns
        """
        packet_sum = 0
        upload_value = CIF[0]
        download_value = CIF[1]
        time_slot_len = len(upload_value)
        real_upload_value = []
        real_download_value = []

        i = 0
        while i * slot_duration <= load_time and i < time_slot_len:
            time_slot_upload, time_slot_download = upload_value[i], download_value[i]
            packet_sum += time_slot_upload + time_slot_download

            if packet_sum <= self.handshake_packet_sum:
                continue

            if random.random() <= self.insert_prob:
                # Insert new time slot
                start = max(0, i - self.num_window_avg)
                end = min(len(download_value), i + self.num_window_avg + 1)
                if end - start >= self.num_window_avg:
                    time_slot_upload = int(np.mean(upload_value[start:end]))
                    time_slot_download = int(np.mean(download_value[start:end]))
                else:
                    time_slot_upload = median_of_non_zero(upload_value)
                    time_slot_download = median_of_non_zero(download_value)
                real_upload_value.append(time_slot_upload)
                real_download_value.append(time_slot_download)
            else:
                # Shrink existing time slot
                if random.random() >= self.shrink_prob:
                    time_slot_upload = int(time_slot_upload * random.uniform(self.shrink_ratio_min, self.shrink_ratio_max))
                    time_slot_download = int(time_slot_download * random.uniform(self.shrink_ratio_min, self.shrink_ratio_max))
                    real_upload_value.append(time_slot_upload)
                    real_download_value.append(time_slot_download)
                i += 1

        # Pad remaining slots with zeros
        for _ in range(time_slot_len - len(real_upload_value)):
            real_upload_value.append(0)
            real_download_value.append(0)

        return np.vstack([np.array(real_upload_value), np.array(real_download_value)])

    def trace_flatten(self, CIF, load_time, slot_duration):
        """
        Trace Flatten: Removes time slots to flatten the traffic pattern.
        
        Args:
            CIF: Consistent Interaction Feature matrix
            load_time: Total load time
            slot_duration: Duration of each time slot
            
        Returns:
            numpy.ndarray: Modified traffic data with flattened pattern
        """
        packet_sum = 0
        upload_value = CIF[0]
        download_value = CIF[1]
        real_upload_value = []
        real_download_value = []

        for i in range(len(upload_value)):
            time_slot_upload, time_slot_download = upload_value[i], download_value[i]
            packet_sum += time_slot_upload + time_slot_download

            if packet_sum <= self.handshake_packet_sum:
                continue

            if i * slot_duration >= load_time:
                break

            if random.random() > self.remove_prob:
                continue

            # Increase packet size
            if random.random() > self.increase_prob:
                time_slot_upload = int(time_slot_upload * (1 + random.uniform(self.increase_ratio_min, self.increase_ratio_max)))
                time_slot_download = int(time_slot_download * (1 + random.uniform(self.increase_ratio_min, self.increase_ratio_max)))
            real_upload_value.append(time_slot_upload)
            real_download_value.append(time_slot_download)

        # Pad remaining slots with zeros
        for _ in range(len(upload_value) - len(real_upload_value)):
            real_upload_value.append(0)
            real_download_value.append(0)

        return np.vstack([np.array(real_upload_value), np.array(real_download_value)])

    def insert_gaussian_noise(self, CIF, load_time, slot_duration):
        """
        Add Gaussian noise to the traffic data.
        
        Args:
            CIF: Consistent Interaction Feature matrix
            load_time: Total load time
            slot_duration: Duration of each time slot
            
        Returns:
            numpy.ndarray: Modified traffic data with added noise
        """
        noise = np.random.normal(loc=0, scale=2, size=CIF.shape)
        end_index = int(load_time / slot_duration)
        CIF[:, :end_index] = CIF[:, :end_index] + noise[:, :end_index]
        return np.round(CIF).astype(int)

    def augment(self, CIF, load_time, slot_duration):
        """
        Apply random augmentation to the traffic data using one of three methods:
        1. Trace Fluctuation: Modifies packet sizes with random fluctuations
        2. Trace Flatten: Removes time slots to flatten the traffic pattern
        3. Trace Aggregation: Inserts new time slots to aggregate traffic patterns
        
        Args:
            CIF: Consistent Interaction Feature matrix
            load_time: Total load time
            slot_duration: Duration of each time slot
            
        Returns:
            numpy.ndarray: Augmented traffic data
        """
        selected_number = random.choice([0, 1, 2])
        if selected_number == 0:
            return self.trace_fluctuation(CIF, load_time, slot_duration)
        elif selected_number == 1:
            return self.trace_aggregation(CIF, load_time, slot_duration)
        else:
            return self.trace_flatten(CIF, load_time, slot_duration)
