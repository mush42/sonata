use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::iter;
use thiserror::Error;

mod inference_engine;
pub use self::inference_engine::DynamicInferenceEngine;

#[cfg(any(feature = "ort", feature = "tract"))]
pub use self::inference_engine::create_inference_engine;

pub type LibtashkeelResult<T> = Result<T, LibtashkeelError>;

pub const CHAR_LIMIT: usize = 12000;
const PAD: char = '_';
static INPUT_ID_MAP: Lazy<HashMap<char, i64>> =
    Lazy::new(|| serde_json::from_str(include_str!("../data/input_id_map.json")).unwrap());
static TARGET_ID_MAP: Lazy<HashMap<u8, String>> = Lazy::new(|| {
    let target_id_map: HashMap<String, u8> =
        serde_json::from_str(include_str!("../data/target_id_map.json")).unwrap();
    reversed_mapping(&target_id_map)
});
static TARGET_META_CHAR_IDS: Lazy<HashSet<u8>> = Lazy::new(|| {
    // Fixme: asumes that input ids are the same as target ids
    HashSet::from_iter([PAD].map(|c| INPUT_ID_MAP[&c]).map(|i| i as u8))
});
static ARABIC_DIACRITICS: Lazy<HashSet<char>> = Lazy::new(|| {
    HashSet::from_iter(
        [1618, 1617, 1614, 1615, 1616, 1611, 1612, 1613]
            .iter()
            .map(|i| unsafe { char::from_u32_unchecked(*i) }),
    )
});
static SENTENCE_BOUNDRIES: Lazy<HashSet<char>> =
    Lazy::new(|| HashSet::from_iter(['.', '،', '؟', '!']));

pub trait InferenceEngine {
    fn infer(
        &self,
        input_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)>;
}

#[derive(Error, Debug)]
pub enum LibtashkeelError {
    #[error("input too long. Expected {0} characters")]
    InputTooLong(usize),
    #[error("Inference error. {0}")]
    InferenceError(String),
    #[error("Resource not found. {0}")]
    ModelLoadError(#[from] std::io::Error),
}

fn reversed_mapping<K, V>(input: &HashMap<K, V>) -> HashMap<V, K>
where
    K: ToOwned<Owned = K>,
    V: ToOwned<Owned = V> + std::hash::Hash + std::cmp::Eq,
{
    HashMap::from_iter(input.iter().map(|(k, v)| (v.to_owned(), k.to_owned())))
}

#[inline(always)]
fn is_diacritic_char(c: char) -> bool {
    ARABIC_DIACRITICS.contains(&c)
}

fn extract_chars_and_diacritics(input_text: &str) -> (String, Vec<String>) {
    let input_text = input_text.trim_start_matches(is_diacritic_char);

    let mut clean_chars = String::new();
    let mut diacritics = Vec::new();

    let mut pending_diac = String::with_capacity(2);
    input_text.chars().chain(iter::once(' ')).for_each(|c| {
        if is_diacritic_char(c) {
            pending_diac.push(c);
        } else {
            clean_chars.push(c);
            diacritics.push(std::mem::take(&mut pending_diac));
        }
    });

    clean_chars.pop().unwrap();
    diacritics.remove(0);
    (clean_chars, diacritics)
}

fn to_valid_chars(input: impl Iterator<Item = char>) -> (String, HashSet<char>) {
    let mut valid = String::new();
    let mut invalid = HashSet::new();
    for c in input {
        if INPUT_ID_MAP.contains_key(&c) {
            valid.push(c);
        } else {
            invalid.insert(c);
        }
    }
    (valid, invalid)
}

fn input_to_ids(input: impl Iterator<Item = char>) -> Vec<i64> {
    Vec::from_iter(input.map(|c| INPUT_ID_MAP[&c]))
}

fn target_to_diacritics(target_ids: impl Iterator<Item = u8>) -> Vec<String> {
    Vec::from_iter(
        target_ids
            .filter(|id| !TARGET_META_CHAR_IDS.contains(id))
            .map(|diac_id| &TARGET_ID_MAP[&diac_id])
            .cloned(),
    )
}

fn annotate_text_with_diacritics(
    input: &str,
    diacritics: Vec<String>,
    removed_chars: HashSet<char>,
) -> String {
    let mut output = String::new();
    let mut diac_iter = diacritics.iter();
    for c in input.chars() {
        if removed_chars.contains(&c) {
            output.push(c);
        } else {
            output.push(c);
            let diac = diac_iter.next().unwrap();
            output.push_str(diac);
        }
    }
    output
}

fn annotate_text_with_diacritics_taskeen(
    input: &str,
    diacritics: Vec<String>,
    removed_chars: HashSet<char>,
    logits: Vec<f32>,
    taskeen_threshold: Option<f32>,
) -> String {
    let taskeen_threshold = taskeen_threshold.unwrap();
    let mut output = String::new();
    let mut diac_iter = diacritics.iter().zip(logits);
    let mut next_char_iter = input.chars();
    next_char_iter.next();
    let sukoon = unsafe { char::from_u32_unchecked(1618u32) };
    for this_char in input.chars() {
        if removed_chars.contains(&this_char) {
            output.push(this_char);
        } else {
            let (diacritic, logit) = diac_iter.next().unwrap();
            output.push(this_char);
            if diacritic.is_empty() {
                continue;
            }
            if logit > taskeen_threshold {
                output.push_str(diacritic);
                continue;
            }
            let next_char = next_char_iter.next();
            if next_char.is_some() && SENTENCE_BOUNDRIES.contains(&next_char.unwrap()) {
                output.push(sukoon);
                continue;
            }
            output.push_str(diacritic);
        }
    }
    output
}

pub fn do_tashkeel(
    engine: &impl InferenceEngine,
    text: &str,
    taskeen_threshold: Option<f32>,
) -> LibtashkeelResult<String> {
    let text = text.trim();

    if text.chars().count() > CHAR_LIMIT {
        return Err(LibtashkeelError::InputTooLong(CHAR_LIMIT));
    }

    let (text, _diacritics) = extract_chars_and_diacritics(text);
    let (input_text, removed_chars) = to_valid_chars(text.chars().take(CHAR_LIMIT));

    let input_ids = input_to_ids(input_text.chars());
    let seq_length = input_ids.len();

    if seq_length > 0 {
        let timer = std::time::Instant::now();
        let (target_ids, logits) = engine.infer(input_ids, seq_length)?;
        let inference_ms = timer.elapsed().as_millis() as f32;
        log::debug!("Inference time: {} ms", inference_ms);
        let diacritics = target_to_diacritics(target_ids.into_iter());
        let final_text = if taskeen_threshold.is_none() {
            annotate_text_with_diacritics(&text, diacritics, removed_chars)
        } else {
            annotate_text_with_diacritics_taskeen(
                &text,
                diacritics,
                removed_chars,
                logits,
                taskeen_threshold,
            )
        };
        Ok(final_text)
    } else {
        log::debug!("Inference time: {} ms", 0.0);
        Ok(text)
    }
}

// ==============================
#[cfg(test)]
mod tests {
    use super::*;

    static INFERENCE_ENGINE: Lazy<DynamicInferenceEngine> =
        Lazy::new(|| create_inference_engine(None).unwrap());

    #[test]
    fn test_extract_diacritics_when_empty() {
        let (chars, diacritics) = extract_chars_and_diacritics("");
        assert_eq!(chars.is_empty(), true);
        assert_eq!(diacritics.is_empty(), true);
    }

    #[test]
    fn test_extract_diacritics() {
        let text = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ";
        let (chars, diacritics) = extract_chars_and_diacritics(text);

        assert_eq!(chars.chars().count(), diacritics.len());

        assert_eq!(chars.chars().nth(0), Some('ب'));
        assert_eq!(diacritics[0], "ِ");

        assert_eq!(chars.chars().nth(6), Some('ل'));
        assert_eq!(diacritics[6], "َّ");
    }

    #[test]
    fn test_basic_tashkeel() -> LibtashkeelResult<()> {
        let text = "بسم الله الرحمن الرحيم";

        let expected = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ";
        let tashkeeled = do_tashkeel(&*INFERENCE_ENGINE, text, None)?;

        assert_ne!(tashkeeled, text);
        assert_eq!(tashkeeled, expected);

        Ok(())
    }
    #[test]
    fn test_taskeen() -> LibtashkeelResult<()> {
        let poem = [
            "من ذا يقارن حسنك المغري بصيف قد تجلى.",
            "وفنون سحرك قد بدت في ناظري أسمى وأغلى.",
            "تجني الرياح العاتيات على البراعم وهي جذلى.",
            "والصيف يمضي مسرعا إذ عقده المحدود ولى.",
        ]
        .join(" ");

        let no_taskeen = do_tashkeel(&*INFERENCE_ENGINE, &poem, None)?;
        let taskeen = do_tashkeel(&*INFERENCE_ENGINE, &poem, Some(0.80))?;

        assert_eq!(taskeen == no_taskeen, false);

        let sukoon = char::from_u32(0x652).unwrap();
        let no_taskeen_sukoon_count = no_taskeen.chars().filter(|c| c == &sukoon).count();
        let taskeen_sukoon_count = taskeen.chars().filter(|c| c == &sukoon).count();
        assert_eq!(taskeen_sukoon_count > no_taskeen_sukoon_count, true);

        Ok(())
    }
}
