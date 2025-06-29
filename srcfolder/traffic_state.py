class TrafficState:
    """
    Manages the current state of traffic at an intersection.
    This includes vehicle counts for each lane and the current signal state
    (e.g., red/green represented by 0/1) for each lane.
    """
    def __init__(self, num_lanes=4):
        """
        Initializes the traffic state.

        Args:
            num_lanes (int): The number of lanes or traffic flow directions to manage.
        """
        self.num_lanes = num_lanes
        # Initialize vehicle count for each lane to zero
        self.lane_counts = [0] * num_lanes
        # Initialize signal state for each lane (0 or 1, e.g., Red/Green or Off/On)
        self.signal_state = [0] * num_lanes

    def update_state(self, lane_counts):
        """
        Updates the vehicle counts for all lanes.

        Args:
            lane_counts (list of int): A list of current vehicle counts for each lane.
                                        Must be of length `num_lanes`.

        Raises:
            ValueError: If the length of `lane_counts` does not match `self.num_lanes`.
        """
        if len(lane_counts) != self.num_lanes:
            raise ValueError(f"Mismatch in lane counts length. Expected {self.num_lanes}, got {len(lane_counts)}")
        self.lane_counts = lane_counts

    def get_state_vector(self):
        """
        Returns a combined state vector consisting of lane vehicle counts
        and signal states. This vector can be used as input to the RL agents.

        Returns:
            list: A concatenated list of `lane_counts` and `signal_state`.
                  Example: [lane1_count, lane2_count, ..., lane1_signal, lane2_signal, ...]
        """
        # Concatenate lane counts and signal states to form a single state vector
        return self.lane_counts + self.signal_state

