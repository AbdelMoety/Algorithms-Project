"""
Backtracking and Visualization of DP Results
Reconstructs optimal segmentation path and prepares for visualization
"""


def reconstruct_segments(backtrack_table, n):
    """
    Reconstruct optimal segmentation from DP backtracking table

    Args:
        backtrack_table (list): List of (position, (prev_pos, seg_type, cost))
        n (int): Total signal length

    Returns:
        list: List of segments with metadata
    """
    if not backtrack_table:
        return []

    # Create dictionary for easier lookup
    bt_dict = {pos: (prev, seg_type, cost) for pos, (prev, seg_type, cost) in backtrack_table}

    # Backtrack from end
    segments = []
    current_pos = n

    while current_pos > 0 and current_pos in bt_dict:
        prev_pos, seg_type, cost = bt_dict[current_pos]

        segment = {
            'type': seg_type,
            'start': prev_pos,
            'end': current_pos - 1,  # Convert to 0-based
            'length': current_pos - prev_pos,
            'cost': float(cost),
            'step': len(segments)  # Step number for animation
        }

        segments.append(segment)
        current_pos = prev_pos

    # Reverse to get chronological order
    segments.reverse()

    # Assign step numbers
    for i, seg in enumerate(segments):
        seg['step'] = i + 1

    return segments


def visualize_backtracking_steps(segments, signal_length):
    """
    Generate step-by-step backtracking visualization data

    Args:
        segments (list): List of segments from DP
        signal_length (int): Total length of signal

    Returns:
        dict: Step-by-step animation data
    """
    steps = []

    # Initial state: no segments identified
    steps.append({
        'step': 0,
        'description': 'Start: No segments identified',
        'segments': [],
        'covered': 0,
        'percentage': 0
    })

    # Add segments one by one
    for i, seg in enumerate(segments):
        current_segments = segments[:i + 1]
        covered = sum(s['length'] for s in current_segments)
        percentage = (covered / signal_length) * 100

        step_desc = {
            'step': i + 1,
            'description': f'Step {i + 1}: Identified {seg["type"]} wave '
                           f'(positions {seg["start"]}-{seg["end"]})',
            'segments': current_segments.copy(),
            'covered': covered,
            'percentage': round(percentage, 1),
            'current_segment': seg
        }
        steps.append(step_desc)

    # Final step: complete segmentation
    steps.append({
        'step': len(segments) + 1,
        'description': f'Complete: All {len(segments)} segments identified',
        'segments': segments,
        'covered': signal_length,
        'percentage': 100,
        'current_segment': None
    })

    return steps