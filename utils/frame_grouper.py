import numpy as np


class FrameGrouper:
    """Optimisation tool: if a set of frames are all considered to be significantly different from the background,
    group them together and select the most different for further processing"""
    def __init__(self):
        self.masks = None
        self.significant_frames = [-3]
        self.frame_scores = []
        self.last_sig_frame = -3

    def append_frame(self, frame_number: int, mask: np.ndarray, frame_score: float) -> None:
        """Add a significant frame to a group. Creates a new group if the last frame was not in a group"""
        if frame_number == self.last_sig_frame + 1:
            # If a group already exists, append to it
            self.significant_frames = [frame_number]
            self.masks = np.expand_dims(mask, axis=2)
            self.frame_scores = [frame_score]
        else:
            # Else, create a new group
            self.significant_frames.append(frame_number)
            self.masks = np.dstack((self.masks, mask))
            self.frame_scores.append(frame_score)
        self.last_sig_frame = frame_number

    def get_best_frame(self) -> np.ndarray:
        """Returns the frame with the highest simm score (most different) from the currently existing group."""
        best_frame_idx = self.frame_scores.index(max(self.frame_scores))
        best_mask = self.masks[:, :, best_frame_idx]
        self.masks = None  # Remove array to save memory
        return best_mask
