use pyo3::prelude::*;
use std::collections::HashMap;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rayon::prelude::*;

fn sample_index_from_weights(weights: Vec<f64>) -> Option<usize> {
    let mut rng = thread_rng();
    sample_index_from_weights_with_rng(&mut rng, weights)
}

fn sample_index_from_weights_with_rng(rng: &mut ThreadRng, weights: Vec<f64>) -> Option<usize> {
    if weights.iter().any(|&w| w < 0.0) {
        return None; // Return None if any weight is negative
    }

    let dist = WeightedIndex::new(&weights).ok()?;
    Some(dist.sample(rng))
}

fn weighted_avg(vec1: &[f64], vec2: &[f64], vec3: &[f64],w1:f64,w2:f64,w3:f64) -> Option<Vec<f64>> {
    if vec1.len() != vec2.len() || vec1.len() != vec3.len() {
        return None;
    }
    let w4:f64;
    if w2 > (w1 + w3) * 5.0{
        w4 = w2/(100.0);
    } else{ 
        w4 = w2;
    }
    Some(
        vec1.iter().zip(vec2).zip(vec3)
        // .map(|((&v1, &v2), &v3)| (v2) / 1.0)
            .map(|((&v1, &v2), &v3)| (v1 * w1 + v2 * w4 + v3 * w3) / 3.0)
            .collect()
    )
}

struct GameResults {
    home_win: bool,
    away_win: bool,
    game_outcome: Vec<String>
}

#[pyfunction] 
fn simulation(games: i32,year:i16,transition:Vec<Vec<f64>>,batter_trans:HashMap<String,Vec<Vec<f64>>>,pitcher_trans:Vec<Vec<Vec<f64>>>,lineups:Vec<Vec<String>>,run_dict:HashMap<String,i8>,outcome_dict:HashMap<String,Vec<String>>,probs_dict:HashMap<String,Vec<f64>>) -> (i32,i32,Vec<String>) {

    let game_results: Vec<GameResults> = (0..games).into_par_iter()
    .map(|i| {
        let mut outcome:Vec<String> = Vec::new();
        let mut inning:f32 = 0.0;
        let mut home_runs:i8 = 0;
        let mut away_runs:i8 = 0;
        let mut home_lineup_idx:i64 = 0;
        let mut away_lineup_idx:i64 = 0;
        let mut batter:String;
        let mut pitcher:i8;
        let home_win:bool;
        let away_win:bool;
        'current_game: loop {
            let mut state:i8;
            if inning >= 10.0 && year > 2019{
                state = 3;
            } else {
                state = 1;
            }

            while state != 25{
                if inning % 1.0 == 0.0{

                    batter = lineups[1][(away_lineup_idx % 9) as usize].to_string();
                    pitcher = 0;
                } else {

                    batter = lineups[0][(home_lineup_idx % 9) as usize].to_string();
                    pitcher = 1;
                }
                let transition_matrix:Vec<Vec<f64>> = batter_trans[&batter].to_vec();

                let pitcher_transition_matrix:Vec<Vec<f64>> = pitcher_trans[pitcher as usize].to_vec();

                let batter_row:&Vec<f64> = &transition_matrix[state as usize - 1];

                let pitcher_row:&Vec<f64> = &pitcher_transition_matrix[state as usize - 1];

                let trans_row:&Vec<f64> = &transition[state as usize - 1];

                let b_sum:f64 = batter_row.iter().sum();

                let p_sum:f64 = pitcher_row.iter().sum();

                let t_sum:f64 = ((b_sum + 1.0) + (p_sum + 1.0)) * 10.0;

                let avg_row:Vec<f64> = weighted_avg(batter_row,trans_row,pitcher_row,b_sum,t_sum,p_sum).unwrap();

                let post_state: i8 = sample_index_from_weights(avg_row.to_vec())
                .unwrap_or(sample_index_from_weights(trans_row.to_vec()).unwrap()) as i8 + 1;
                let trans_string = format!("{}{}", state, post_state);
                

                let runs_scored:i8 = run_dict[&trans_string];

                let outcome_row:&Vec<String> = &outcome_dict[&trans_string];

                let outcome_probs:&Vec<f64> = &probs_dict[&trans_string];

                let outcome_idx:i8 = sample_index_from_weights(outcome_probs.to_vec()).unwrap() as i8;

                let play_outcome:&String = &outcome_row[outcome_idx as usize];

                let outcome_str:String = format!("{}${}${}${}${}${}", batter, play_outcome, state, post_state, inning, i);
                
                outcome.push(outcome_str);

                if inning % 1.0 == 0.0{
                    away_runs += runs_scored;
                    away_lineup_idx += 1;
                } else {
                    home_runs += runs_scored;
                    home_lineup_idx += 1;
                }
                if (home_runs > away_runs && inning > 9.0) || (home_runs < away_runs && inning > 9.0 && post_state == 25){
                    break 'current_game;
                }
                state = post_state;
            }
            inning += 0.5;
        }
        if away_runs > home_runs{
            away_win = true;
            home_win = false;
        } else {
            home_win = true;
            away_win = false;
        }
        GameResults { home_win: home_win, away_win: away_win,game_outcome: outcome}
    }).collect();
    // Aggregate the results
    let home_wins:i32 = game_results.iter().filter(|result| result.home_win).count() as i32;
    let away_wins:i32 = game_results.iter().filter(|result| result.away_win).count() as i32;
    let game_outcomes:Vec<String> = game_results.iter().map(|result| &result.game_outcome).flatten().map(|result| result.to_string()).collect::<Vec<_>>();

    (home_wins,away_wins,game_outcomes)
}


#[pymodule]
fn game_simulations(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulation, m)?)?;
    Ok(())
}