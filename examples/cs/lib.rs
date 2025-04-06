use std::collections::HashMap;

use pyo3::{
    pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};

#[pyfunction]
pub fn count_characters(text: String) -> HashMap<String, usize> {
    let mut char_count = HashMap::new();
    for c in text.chars() {
        let count = char_count.entry(c.to_string()).or_insert(0);
        *count += 1;
    }
    char_count
}

#[pymodule(name = "cs_lib")]
pub fn py_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_characters, m)?)?;
    Ok(())
}
