pub mod helpers;
pub mod test_board_hardening;
pub mod test_movegen_hardening;
pub mod test_game_logic_hardening;
pub mod test_agent_hardening;
pub mod test_training_hardening;
pub mod test_determinism_hardening;

pub mod property {
    pub mod test_board_props;
    pub mod test_movegen_props;
    pub mod test_agent_props;
    pub mod test_determinism_props;
}

pub mod fuzz {
    pub mod test_chess_fuzz;
}
