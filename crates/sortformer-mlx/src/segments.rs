//! Threshold-based conversion of frame probabilities into speaker segments.

/// A contiguous span of one speaker's activity.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SpeakerSegment {
    /// Zero-based Sortformer output slot.
    pub speaker: u8,
    /// Segment start in milliseconds.
    pub start_ms: u64,
    /// Exclusive segment end in milliseconds.
    pub end_ms: u64,
}

/// Binarizes per-speaker activity and merges it into stable segments.
pub fn segments_from_probs(
    probs: &[[f32; 4]],
    threshold: f32,
    min_frames: usize,
    frame_ms: u64,
) -> Vec<SpeakerSegment> {
    let mut segments = Vec::new();
    for speaker in 0..4u8 {
        let mut active: Vec<bool> = probs
            .iter()
            .map(|frame| frame[speaker as usize] >= threshold)
            .collect();
        close_short_gaps(&mut active, min_frames);
        drop_short_islands(&mut active, min_frames);
        let mut start = None;
        for (frame, &on) in active.iter().enumerate() {
            match (on, start) {
                (true, None) => start = Some(frame),
                (false, Some(begin)) => {
                    segments.push(segment(speaker, begin, frame, frame_ms));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = start {
            segments.push(segment(speaker, begin, active.len(), frame_ms));
        }
    }
    segments.sort_by_key(|segment| (segment.start_ms, segment.speaker));
    segments
}

fn segment(speaker: u8, start: usize, end: usize, frame_ms: u64) -> SpeakerSegment {
    SpeakerSegment {
        speaker,
        start_ms: start as u64 * frame_ms,
        end_ms: end as u64 * frame_ms,
    }
}

fn close_short_gaps(active: &mut [bool], min_frames: usize) {
    let mut frame = 0;
    while frame < active.len() {
        if !active[frame] {
            let gap_start = frame;
            while frame < active.len() && !active[frame] {
                frame += 1;
            }
            let bounded = gap_start > 0 && frame < active.len();
            if bounded && frame - gap_start < min_frames {
                active[gap_start..frame].fill(true);
            }
        } else {
            frame += 1;
        }
    }
}

fn drop_short_islands(active: &mut [bool], min_frames: usize) {
    let mut frame = 0;
    while frame < active.len() {
        if active[frame] {
            let island_start = frame;
            while frame < active.len() && active[frame] {
                frame += 1;
            }
            if frame - island_start < min_frames {
                active[island_start..frame].fill(false);
            }
        } else {
            frame += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs(rows: &[[f32; 4]]) -> Vec<[f32; 4]> {
        rows.to_vec()
    }

    #[test]
    fn continuous_activity_becomes_one_segment() {
        let input = probs(&[[0.9, 0.0, 0.0, 0.0]; 10]);
        let segments = segments_from_probs(&input, 0.5, 2, 80);
        assert_eq!(
            segments,
            vec![SpeakerSegment { speaker: 0, start_ms: 0, end_ms: 800 }]
        );
    }

    #[test]
    fn short_islands_are_dropped_and_short_gaps_closed() {
        let mut input = vec![[0.0f32; 4]; 20];
        for frame in 0..8 {
            input[frame][1] = 0.9; // speaker 1 active 0..8
        }
        input[9][1] = 0.9; // 1-frame island after a 1-frame gap: gap closes
        for frame in 15..16 {
            input[frame][2] = 0.9; // 1-frame island for speaker 2: dropped
        }
        let segments = segments_from_probs(&input, 0.5, 2, 80);
        assert_eq!(
            segments,
            vec![SpeakerSegment { speaker: 1, start_ms: 0, end_ms: 800 }]
        );
    }

    #[test]
    fn overlapping_speakers_yield_overlapping_segments() {
        let mut input = vec![[0.0f32; 4]; 10];
        for frame in 0..10 {
            input[frame][0] = 0.8;
        }
        for frame in 5..10 {
            input[frame][3] = 0.8;
        }
        let segments = segments_from_probs(&input, 0.5, 2, 80);
        assert_eq!(
            segments,
            vec![
                SpeakerSegment { speaker: 0, start_ms: 0, end_ms: 800 },
                SpeakerSegment { speaker: 3, start_ms: 400, end_ms: 800 },
            ]
        );
    }
}
