use nemotron_mlx::model::{AudioChunkScheduler, EncoderConfig, StreamingChunkPlan};

fn scheduler() -> AudioChunkScheduler {
    let plan = StreamingChunkPlan::new(&EncoderConfig::nemotron_3_5(), 3).unwrap();
    AudioChunkScheduler::new(plan)
}

#[test]
fn waits_for_complete_first_and_subsequent_windows() {
    let mut scheduler = scheduler();

    assert!(scheduler.push(&vec![1.0; 4_039]).unwrap().is_empty());
    let first = scheduler.push(&[2.0]).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].samples.len(), 4_040);
    assert!(first[0].center);
    assert_eq!(first[0].mel_frames, 25);

    assert!(scheduler.push(&vec![3.0; 5_223]).unwrap().is_empty());
    let second = scheduler.push(&[4.0]).unwrap();
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].samples.len(), 5_520);
    assert!(!second[0].center);
    assert_eq!(second[0].mel_frames, 32);
    assert_eq!(&second[0].samples[..295], &vec![1.0; 295][..]);
    assert_eq!(second[0].samples[295], 2.0);
}

#[test]
fn finish_pads_at_most_one_tail_and_locks_the_scheduler() {
    let mut scheduler = scheduler();
    scheduler.push(&vec![1.0; 4_040]).unwrap();

    let tail = scheduler.finish().unwrap().expect("overlapping tail");
    assert_eq!(tail.samples.len(), 5_520);
    assert!(!tail.center);
    assert_eq!(tail.samples[..296], vec![1.0; 296]);
    assert!(tail.samples[296..].iter().all(|sample| *sample == 0.0));

    assert!(scheduler.finish().is_err());
    assert!(scheduler.push(&[1.0]).is_err());
}

#[test]
fn reset_clears_audio_and_lifecycle_state() {
    let mut scheduler = scheduler();
    scheduler.push(&vec![1.0; 4_040]).unwrap();
    scheduler.finish().unwrap();
    scheduler.reset();

    assert!(scheduler.push(&vec![2.0; 4_039]).unwrap().is_empty());
    let first = scheduler.push(&[3.0]).unwrap();
    assert_eq!(first.len(), 1);
    assert!(first[0].center);
    assert_eq!(first[0].samples[4_039], 3.0);
}

#[test]
fn finishing_empty_audio_emits_nothing() {
    let mut scheduler = scheduler();
    assert!(scheduler.finish().unwrap().is_none());
}
