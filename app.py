def render_residential_tab(calculator):
    """Render residential units input tab"""
    st.header("🏢 Residential Units")
    
    # Initialize residential counts in session state if not present
    if 'residential_counts' not in st.session_state:
        st.session_state.residential_counts = {unit_type: 0 for unit_type in calculator.residential_units.keys()}
    
    col1, col2 = st.columns(2)
    
    unit_counts = {}
    
    with col1:
        st.subheader("Unit Types")
        for i, (unit_type, load) in enumerate(list(calculator.residential_units.items())[:3]):
            st.session_state.residential_counts[unit_type] = st.number_input(
                f"{unit_type} ({load} kVA)",
                min_value=0,
                value=st.session_state.residential_counts[unit_type],
                step=1,
                key=f"res_{unit_type}"  # Use a unique key based on unit_type
            )
            unit_counts[unit_type] = st.session_state.residential_counts[unit_type]
    
    with col2:
        st.subheader("Unit Types (cont.)")
        for i, (unit_type, load) in enumerate(list(calculator.residential_units.items())[3:]):
            st.session_state.residential_counts[unit_type] = st.number_input(
                f"{unit_type} ({load} kVA)",
                min_value=0,
                value=st.session_state.residential_counts[unit_type],
                step=1,
                key=f"res_{unit_type}"
            )
            unit_counts[unit_type] = st.session_state.residential_counts[unit_type]
    
    # Sample data from Excel
    with st.expander("📋 Load Sample Data"):
        if st.button("Load Sample Residential Data"):
            # Update the dictionary
            st.session_state.residential_counts = {
                "Studio Apartment (1-room)": 0,
                "Studio Apartment (2-room)": 0,
                "2-room flat": 68,
                "3-room flat": 0,
                "4-room flat": 0,
                "5-room flat": 515
            }
            st.success("Sample data loaded!")
            st.rerun()
    
    return unit_counts
