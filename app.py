# app.py - Final version with all features and fixed sample button

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import io

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RetailShop:
    name: str
    area_sqm: float
    load_density_w_per_sqm: float
    breaker_size: str
    power_factor: float = 0.85

    @property
    def load_kw(self) -> float:
        return (self.area_sqm * self.load_density_w_per_sqm) / 1000

    @property
    def load_kva(self) -> float:
        return self.load_kw / self.power_factor


@dataclass
class ContainmentItem:
    description: str
    cable_size_mm2: str
    quantity: int
    cable_outer_diameter_mm: float


@dataclass
class VoltageDrop:
    percentage: float
    is_compliant: bool
    recommended_cable: str


# ============================================================================
# MAIN CALCULATOR CLASS
# ============================================================================

class ElectricalLoadCalculator:
    def __init__(self):
        self.residential_units = {
            "Studio Apartment (1-room)": 3.5,
            "Studio Apartment (2-room)": 3.5,
            "2-room flat": 3.5,
            "3-room flat": 3.8,
            "4-room flat": 4.3,
            "5-room flat": 4.5
        }

        self.common_installations = {
            "Escalator": 22,
            "Service Lift (20 man)": 35,
            "Lift 11 sty (13 man)": 20,
            "Lift (20 man)": 35,
            "Public Lighting Circuit": 12,
            "Outdoor Lighting Circuit": 17,
            "Domestic Booster Pump": 3.52,
            "Fire Hose Reel Pump": 2.6,
            "Refuse Handling Plant": 10,
            "Wet Riser System": 25,
            "Sprinkler System": 44,
            "Mech Ventilation": 11
        }

        self.facilities = {
            "Future Communal Facilities": 42,
            "Bin Centre": 10,
            "Shop (per 100sqm)": 42,
            "Hawker Centre": 676.88,
            "RC Centre": 42,
            "Polyclinic": 881.28,
            "MNO": 41.6,
            "EPS": 14,
            "KDC": 104,
            "CC": 693,
            "Mech": 762
        }

        # Retail load densities (W/m²)
        self.retail_load_densities = {
            "General Retail": 120,
            "Restaurant/Café": 250,
            "Supermarket": 200,
            "Pharmacy/Medical": 150,
            "Fashion/Boutique": 100,
            "Electronics": 150,
            "Furniture": 80,
            "Gym/Fitness": 180,
            "Food Court": 220,
            "Convenience Store": 180,
            "Chinese Medicine": 168,
            "Halal Café": 228,
            "Takeaway Food": 285
        }

        self.diversity_factors = {
            "residential": 0.8,
            "commercial": 0.9,
            "common_services": 0.7,
            "lift": 0.5,
            "pump": 0.8,
            "lighting": 0.9,
            "power": 0.8,
            "emergency": 1.0,
            "retail": 0.85,
            "hawker": 0.8
        }

        # Cable current capacity (mm² -> A)
        self.cable_current_capacity = {
            "1.5": 17.5, "2.5": 24, "4": 32, "6": 41, "10": 57,
            "16": 76, "25": 101, "35": 125, "50": 151, "70": 192,
            "95": 232, "120": 269, "150": 300, "185": 341, "240": 400,
            "300": 458
        }

        # Cable outer diameter (mm)
        self.cable_outer_diameter = {
            "1.5": 8.0, "2.5": 8.8, "4": 9.8, "6": 10.9, "10": 12.8,
            "16": 14.9, "25": 18.2, "35": 20.5, "50": 23.4, "70": 26.5,
            "95": 29.8, "120": 32.6, "150": 35.8, "185": 39.2, "240": 44.5,
            "300": 49.0
        }

        # Standard containment sizes (width x height in mm)
        self.standard_containment_sizes = [
            (50, 50), (75, 50), (100, 50), (100, 75), (100, 100),
            (150, 50), (150, 75), (150, 100), (150, 150),
            (200, 50), (200, 75), (200, 100), (200, 150), (200, 200),
            (300, 100), (300, 150), (300, 200), (300, 300),
            (400, 100), (400, 150), (400, 200), (400, 300), (400, 400),
            (500, 100), (500, 150), (500, 200), (500, 300), (500, 400), (500, 500)
        ]

    # ---------- Calculation Methods ----------
    def calculate_residential_load(self, unit_counts: Dict[str, int]) -> float:
        total = 0
        for unit_type, count in unit_counts.items():
            if count > 0 and unit_type in self.residential_units:
                total += self.residential_units[unit_type] * count
        return total

    def calculate_common_load(self, installation_counts: Dict[str, int]) -> float:
        total = 0
        for inst, count in installation_counts.items():
            if count > 0 and inst in self.common_installations:
                total += self.common_installations[inst] * count
        return total

    def calculate_retail_load(self, shops: List[RetailShop], apply_diversity: bool = True) -> Dict:
        if not shops:
            return {"total_kw": 0, "total_kva": 0, "diversity_factor": 1.0,
                    "demand_kw": 0, "demand_kva": 0, "shops": shops}
        total_kw = sum(s.load_kw for s in shops)
        total_kva = sum(s.load_kva for s in shops)
        div = self.diversity_factors["retail"] if apply_diversity else 1.0
        return {
            "total_kw": total_kw,
            "total_kva": total_kva,
            "diversity_factor": div,
            "demand_kw": total_kw * div,
            "demand_kva": total_kva * div,
            "shops": shops
        }

    def calculate_hawker_centre(self, area: float, stalls: int = 40,
                                light_density: float = 20, mv_motors: int = 8,
                                mv_power: float = 11, office_ac: float = 3) -> Dict:
        cooked = stalls * 8
        lighting = area * light_density / 1000
        mv = mv_motors * mv_power
        total_kw = cooked + lighting + mv + office_ac
        total_kva = total_kw / 0.85
        div = self.diversity_factors["hawker"]
        return {
            "cooked_food_load_kw": cooked,
            "lighting_load_kw": lighting,
            "mech_vent_load_kw": mv,
            "office_ac_kw": office_ac,
            "total_kw": total_kw,
            "total_kva": total_kva,
            "diversity_factor": div,
            "demand_kva": total_kva * div
        }

    def calculate_voltage_drop(self, current_a: float, length_m: float,
                               cable_size: str, pf: float = 0.85,
                               voltage_v: int = 400) -> VoltageDrop:
        # Resistance and reactance per km (copper)
        cable_data = {
            "1.5": {"r": 14.8, "x": 0.155}, "2.5": {"r": 8.91, "x": 0.145},
            "4": {"r": 5.57, "x": 0.135}, "6": {"r": 3.71, "x": 0.13},
            "10": {"r": 2.24, "x": 0.125}, "16": {"r": 1.41, "x": 0.12},
            "25": {"r": 0.89, "x": 0.115}, "35": {"r": 0.67, "x": 0.11},
            "50": {"r": 0.49, "x": 0.105}, "70": {"r": 0.35, "x": 0.1},
            "95": {"r": 0.26, "x": 0.095}, "120": {"r": 0.21, "x": 0.09},
            "150": {"r": 0.17, "x": 0.085}, "185": {"r": 0.14, "x": 0.08},
            "240": {"r": 0.11, "x": 0.075}, "300": {"r": 0.09, "x": 0.07}
        }
        if cable_size not in cable_data:
            return VoltageDrop(999, False, "Unknown")
        d = cable_data[cable_size]
        sin_phi = math.sqrt(1 - pf ** 2)
        vd_v = math.sqrt(3) * current_a * (length_m / 1000) * (d["r"] * pf + d["x"] * sin_phi)
        vd_pct = (vd_v / voltage_v) * 100
        compliant = vd_pct <= 4.0
        # Recommend larger cable if needed
        recommended = cable_size
        if not compliant:
            for sz in sorted([float(s) for s in cable_data.keys()]):
                if float(sz) > float(cable_size):
                    test = self.calculate_voltage_drop(current_a, length_m, str(int(sz)), pf, voltage_v)
                    if test.percentage <= 4.0:
                        recommended = str(int(sz))
                        break
        return VoltageDrop(vd_pct, compliant, recommended)

    def calculate_fault_current(self, tx_kva: float, imp_pct: float = 5.0, volt_v: int = 400) -> Dict:
        flc = tx_kva * 1000 / (math.sqrt(3) * volt_v)
        fault_ka = (flc * 100) / imp_pct / 1000
        peak_ka = fault_ka * 2.5
        return {
            "full_load_current_a": flc,
            "fault_current_ka": fault_ka,
            "peak_current_ka": peak_ka
        }

    def calculate_pfc(self, load_kw: float, cur_pf: float, tar_pf: float = 0.95) -> Dict:
        cur_kvar = load_kw * math.tan(math.acos(cur_pf))
        tar_kvar = load_kw * math.tan(math.acos(tar_pf))
        req_kvar = cur_kvar - tar_kvar
        savings = ((cur_pf - tar_pf) / cur_pf) * 100
        return {
            "current_kvar": cur_kvar,
            "target_kvar": tar_kvar,
            "required_capacitor_kvar": req_kvar,
            "estimated_savings_pct": savings
        }

    def calculate_containment(self, items: List[ContainmentItem], fill_ratio: float = 0.4) -> Dict:
        if not items:
            return {"total_area_mm2": 0, "required_area_mm2": 0, "recommended_size": "None", "fill_percentage": 0, "breakdown": []}
        total_area = 0
        breakdown = []
        for it in items:
            if it.quantity > 0 and it.cable_size_mm2 in self.cable_outer_diameter:
                dia = self.cable_outer_diameter[it.cable_size_mm2]
                area_per = math.pi * (dia / 2) ** 2
                item_area = area_per * it.quantity
                total_area += item_area
                breakdown.append({
                    "Description": it.description,
                    "Cable Size (mm²)": it.cable_size_mm2,
                    "Quantity": it.quantity,
                    "Area per Cable (mm²)": round(area_per, 2),
                    "Total Area (mm²)": round(item_area, 2)
                })
        required = total_area / fill_ratio
        rec_size = None
        for w, h in sorted(self.standard_containment_sizes, key=lambda x: x[0] * x[1]):
            if w * h >= required:
                rec_size = f"{w} x {h} mm"
                break
        if rec_size is None:
            rec_size = "Exceeds standard sizes, custom required"
        fill_pct = 0
        if rec_size != "Exceeds standard sizes, custom required":
            w, h = map(int, rec_size.replace(" mm", "").split(" x "))
            fill_pct = (total_area / (w * h)) * 100
        return {
            "total_area_mm2": total_area,
            "required_area_mm2": required,
            "fill_ratio": fill_ratio,
            "recommended_size": rec_size,
            "fill_percentage": fill_pct,
            "breakdown": breakdown
        }


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100.png?text=HDB+Logo", use_column_width=True)
        st.title("Project Information")
        proj = st.text_input("Project Title", "PROPOSED PUBLIC HOUSING DEVELOPMENT")
        ref = st.text_input("Project Reference No.", "Axxx")
        loc = st.text_input("Location Description", "CCK")
        st.divider()
        st.subheader("Professional Engineer")
        pe_name = st.text_input("Name", "Ting Ik Hing")
        pe_reg = st.text_input("Registration No.", "3348")
        firm = st.text_input("Firm Name", "Surbana International Consultants Pte Ltd")
        tel = st.text_input("Telephone No.", "62481315")
        st.divider()
        st.date_input("Date", datetime.now())
        return {"project_title": proj, "project_ref": ref, "location": loc,
                "pe_name": pe_name, "pe_reg_no": pe_reg, "firm_name": firm, "telephone": tel}


def render_residential_tab(calc):
    st.header("🏢 Residential Units")
    if 'residential_counts' not in st.session_state:
        st.session_state.residential_counts = {k: 0 for k in calc.residential_units.keys()}
    col1, col2 = st.columns(2)
    unit_counts = {}
    with col1:
        st.subheader("Unit Types")
        for ut, load in list(calc.residential_units.items())[:3]:
            st.session_state.residential_counts[ut] = st.number_input(
                f"{ut} ({load} kVA)", min_value=0,
                value=st.session_state.residential_counts[ut], step=1
            )
            unit_counts[ut] = st.session_state.residential_counts[ut]
    with col2:
        st.subheader("Unit Types (cont.)")
        for ut, load in list(calc.residential_units.items())[3:]:
            st.session_state.residential_counts[ut] = st.number_input(
                f"{ut} ({load} kVA)", min_value=0,
                value=st.session_state.residential_counts[ut], step=1
            )
            unit_counts[ut] = st.session_state.residential_counts[ut]
    with st.expander("📋 Load Sample Data"):
        if st.button("Load Sample Residential Data"):
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


def render_common_tab(calc):
    st.header("⚙️ Common Services")
    inst = {}
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lifts & Escalators")
        inst["Escalator"] = st.number_input("Escalator (22 kVA)", 0, 10, 2, key="com_esc")
        inst["Service Lift (20 man)"] = st.number_input("Service Lift (20 man, 35 kVA)", 0, 10, 1, key="com_sl")
        inst["Lift 11 sty (13 man)"] = st.number_input("Lift 11 sty (13 man, 20 kVA)", 0, 10, 1, key="com_l11")
        inst["Lift (20 man)"] = st.number_input("Lift (20 man, 35 kVA)", 0, 10, 0, key="com_l20")
    with c2:
        st.subheader("Pumps & Lighting")
        inst["Domestic Booster Pump"] = st.number_input("Domestic Booster Pump (3.52 kVA)", 0, 10, 0, key="com_dbp")
        inst["Fire Hose Reel Pump"] = st.number_input("Fire Hose Reel Pump (2.6 kVA)", 0, 10, 1, key="com_fhr")
        inst["Public Lighting Circuit"] = st.number_input("Public Lighting Circuit (12 kVA)", 0, 10, 1, key="com_pl")
        inst["Outdoor Lighting Circuit"] = st.number_input("Outdoor Lighting Circuit (17 kVA)", 0, 10, 1, key="com_ol")
    st.subheader("Fire Protection Systems")
    c3, c4 = st.columns(2)
    with c3:
        inst["Sprinkler System"] = st.number_input("Sprinkler System (44 kVA)", 0, 10, 0, key="com_spr")
        inst["Wet Riser System"] = st.number_input("Wet Riser System (25 kVA)", 0, 10, 0, key="com_wr")
    with c4:
        inst["Refuse Handling Plant"] = st.number_input("Refuse Handling Plant (10 kVA)", 0, 10, 1, key="com_rh")
        inst["Mech Ventilation"] = st.number_input("Mech Ventilation (11 kVA)", 0, 20, 0, key="com_mv")
    return inst


def render_facilities_tab(calc):
    st.header("🏛️ Facilities")
    fac = {}
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Community Facilities")
        fac["Future Communal Facilities"] = st.number_input("Future Communal Facilities (42 kVA)", 0, 1000, 0, key="fac_fcf")
        fac["Bin Centre"] = st.number_input("Bin Centre (10 kVA)", 0, 100, 0, key="fac_bin")
        fac["RC Centre"] = st.number_input("RC Centre (42 kVA)", 0, 100, 0, key="fac_rc")
        fac["CC"] = st.number_input("CC (693 kVA)", 0, 2000, 0, key="fac_cc")
    with c2:
        st.subheader("Healthcare")
        fac["Polyclinic"] = st.number_input("Polyclinic (881.28 kVA)", 0, 2000, 0, key="fac_poly")
        fac["KDC"] = st.number_input("KDC (104 kVA)", 0, 500, 0, key="fac_kdc")
        st.subheader("Others")
        fac["Hawker Centre"] = st.number_input("Hawker Centre (676.88 kVA/3500sqm)", 0, 2000, 0, key="fac_hawker")
        fac["MNO"] = st.number_input("MNO (41.6 kVA)", 0, 500, 0, key="fac_mno")
        fac["EPS"] = st.number_input("EPS (14 kVA)", 0, 500, 0, key="fac_eps")
        fac["Mech"] = st.number_input("Mech (762 kVA)", 0, 2000, 0, key="fac_mech")
    return fac


def render_retail_tab(calc):
    st.header("🛍️ Retail Shops Details")
    st.info("Load = Area (m²) × Load Density (W/m²)")
    sample = [
        {"name": "Retail (Chinese medicine)", "area": 86.3, "load_density": 168, "breaker": "63A SPN"},
        {"name": "Convenience Store", "area": 121.28, "load_density": 228, "breaker": "40A TPN"},
        {"name": "Open Retail", "area": 27.98, "load_density": 518, "breaker": "63A SPN"},
        {"name": "Halal Café", "area": 121.34, "load_density": 228, "breaker": "40A TPN"},
        {"name": "Takeaway food", "area": 50.85, "load_density": 285, "breaker": "63A SPN"},
        {"name": "Gym", "area": 299.93, "load_density": 185, "breaker": "80A TPN"}
    ]
    if 'shops' not in st.session_state:
        st.session_state.shops = sample.copy()
    df = pd.DataFrame(st.session_state.shops)
    df['load_kw'] = (df['area'] * df['load_density']) / 1000
    display = df[['name', 'area', 'load_density', 'breaker', 'load_kw']].rename(
        columns={'name': 'Shop Name', 'area': 'Area (m²)', 'load_density': 'Load Density (W/m²)',
                 'breaker': 'Breaker Size', 'load_kw': 'Load (kW)'})
    edited = st.data_editor(
        display,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Shop Name": st.column_config.TextColumn("Shop Name", required=True),
            "Area (m²)": st.column_config.NumberColumn("Area (m²)", min_value=0, format="%.2f", required=True),
            "Load Density (W/m²)": st.column_config.NumberColumn("Load Density (W/m²)", min_value=0, format="%.0f", required=True),
            "Breaker Size": st.column_config.TextColumn("Breaker Size", required=True),
            "Load (kW)": st.column_config.NumberColumn("Load (kW)", format="%.3f", disabled=True)
        }
    )
    if not edited.empty:
        updated = []
        for _, row in edited.iterrows():
            updated.append({
                "name": row["Shop Name"],
                "area": row["Area (m²)"],
                "load_density": row["Load Density (W/m²)"],
                "breaker": row["Breaker Size"]
            })
        st.session_state.shops = updated
        calc_df = pd.DataFrame(updated)
        calc_df['Load (kW)'] = (calc_df['area'] * calc_df['load_density']) / 1000
        calc_df['Load (kVA)'] = calc_df['Load (kW)'] / 0.85
        calc_df['Current (A)'] = calc_df['Load (kW)'] * 1000 / (1.732 * 400 * 0.85)
        st.subheader("Calculated Shop Loads")
        st.dataframe(calc_df[['name', 'area', 'load_density', 'Load (kW)', 'Load (kVA)', 'Current (A)']].round(2),
                     use_container_width=True)
        shops = []
        for s in updated:
            try:
                shops.append(RetailShop(
                    name=s["name"], area_sqm=s["area"],
                    load_density_w_per_sqm=s["load_density"],
                    breaker_size=s["breaker"], power_factor=0.85
                ))
            except Exception:
                pass
        return shops
    return []


def render_hawker_tab():
    st.header("🍜 Hawker Centre Details")
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sqm)", 0, 10000, 3500, key="hawk_area")
        stalls = st.number_input("Number of Cooked Food Stalls", 0, 100, 40, key="hawk_stalls")
        light_den = st.number_input("Lighting Density (W/m²)", 0.0, 50.0, 20.0, key="hawk_light")
    with c2:
        mv_motors = st.number_input("Number of MV Motors", 0, 20, 8, key="hawk_mv")
        mv_power = st.number_input("MV Motor Power (kW)", 0.0, 50.0, 11.0, key="hawk_mvp")
        office_ac = st.number_input("Management Office AC (kW)", 0.0, 20.0, 3.0, key="hawk_ac")
        div = st.slider("Diversity Factor", 0.5, 1.0, 0.8, 0.05, key="hawk_div")
    cooked = stalls * 8
    lighting = area * light_den / 1000
    mv = mv_motors * mv_power
    total_kw = cooked + lighting + mv + office_ac
    total_kva = total_kw / 0.85
    st.subheader("Hawker Centre Load Summary")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Connected Load", f"{total_kw:.2f} kW")
    mc2.metric("Connected Load", f"{total_kva:.2f} kVA")
    mc3.metric("Demand Load (with diversity)", f"{total_kva * div:.2f} kVA")
    return {"area": area, "stalls": stalls, "total_kw": total_kw, "total_kva": total_kva, "demand_kva": total_kva * div}


def render_distribution_tab():
    st.header("🔌 Distribution System")
    data = {
        "PG Incoming": ["PG Incoming 1", "PG Incoming 2", "PG Incoming 3", "PG Incoming 4"],
        "MSB": ["MSB 1", "MSB 1", "MSB 2", "MSB 2"],
        "SSB": ["SSB-Resi", "SSB-PC", "SSB-CC", "SSB-HWC"],
        "Other": ["SSB-Shop, Chiller Plant", "SSB-KDC", "EMSB-1", "EMSB-2"],
        "Load (kVA)": [1067, 985, 951, 925]
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)
    # Simple diagram
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 2, 4, 6, 8], y=[5, 5, 5, 5, 5],
        mode='markers+text',
        marker=dict(size=[40, 30, 30, 30, 30], color=['darkblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']),
        text=['Source', 'PG1', 'PG2', 'PG3', 'PG4'],
        textposition="middle center", textfont=dict(color='white', size=12)
    ))
    loads = ['Resi (1067)', 'PC (985)', 'CC (951)', 'HWC (925)']
    for i, ld in enumerate(loads):
        fig.add_trace(go.Scatter(
            x=[2 + i*2, 2 + i*2], y=[4, 3],
            mode='lines+markers+text',
            line=dict(color='gray', width=2),
            marker=dict(size=20, color='orange'),
            text=['', ld], textposition="bottom center",
            showlegend=False
        ))
    fig.update_layout(title="Single Line Diagram", showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_voltage_drop_tab(calc):
    st.header("⚡ Voltage Drop Calculator")
    c1, c2 = st.columns(2)
    with c1:
        cur = st.number_input("Current (A)", 0.0, value=100.0, step=10.0, key="vd_cur")
        length = st.number_input("Cable Length (m)", 0.0, value=50.0, step=5.0, key="vd_len")
        volt = st.selectbox("System Voltage (V)", [230, 400, 690], index=1, key="vd_volt")
    with c2:
        sizes = list(calc.cable_current_capacity.keys())
        cable = st.selectbox("Cable Size (mm²)", sizes, index=10, key="vd_cable")
        pf = st.slider("Power Factor", 0.7, 1.0, 0.85, 0.01, key="vd_pf")
    if st.button("Calculate Voltage Drop", key="vd_btn"):
        res = calc.calculate_voltage_drop(cur, length, cable, pf, volt)
        cc, cstat, crec = st.columns(3)
        cc.metric("Voltage Drop", f"{res.percentage:.2f}%")
        cstat.metric("Status", "✅ Compliant" if res.is_compliant else "❌ Not Compliant")
        if not res.is_compliant:
            crec.metric("Recommended Cable", f"{res.recommended_cable} mm²")
        cap = calc.cable_current_capacity.get(cable, 0)
        if cur > cap:
            st.error(f"⚠️ Current ({cur}A) exceeds cable capacity ({cap}A)")


def render_fault_current_tab(calc):
    st.header("⚡ Fault Current Calculator")
    c1, c2 = st.columns(2)
    with c1:
        tx = st.number_input("Transformer Rating (kVA)", 100, value=1000, step=100, key="fc_tx")
        imp = st.number_input("Transformer Impedance (%)", 1.0, value=5.0, step=0.5, format="%.1f", key="fc_imp")
    with c2:
        volt = st.selectbox("Secondary Voltage (V)", [400, 690], index=0, key="fc_volt")
    if st.button("Calculate Fault Current", key="fc_btn"):
        res = calc.calculate_fault_current(tx, imp, volt)
        col1, col2, col3 = st.columns(3)
        col1.metric("Full Load Current", f"{res['full_load_current_a']:.0f} A")
        col2.metric("Fault Current", f"{res['fault_current_ka']:.2f} kA")
        col3.metric("Peak Current", f"{res['peak_current_ka']:.2f} kA")
        st.subheader("Recommended Breaker Ratings")
        df = pd.DataFrame({
            "Breaker Type": ["MCCB", "ACB"],
            "Rated Current (A)": [res['full_load_current_a'] * 1.25] * 2,
            "Breaking Capacity (kA)": [res['fault_current_ka'] * 1.1] * 2
        })
        st.dataframe(df, use_container_width=True)


def render_pfc_tab(calc):
    st.header("⚡ Power Factor Correction")
    c1, c2 = st.columns(2)
    with c1:
        kw = st.number_input("Load (kW)", 0.0, value=500.0, step=50.0, key="pfc_kw")
        cur_pf = st.slider("Current Power Factor", 0.5, 0.95, 0.8, 0.01, key="pfc_cur")
    with c2:
        tar_pf = st.slider("Target Power Factor", 0.85, 1.0, 0.95, 0.01, key="pfc_tar")
    if st.button("Calculate Capacitor Bank", key="pfc_btn"):
        res = calc.calculate_pfc(kw, cur_pf, tar_pf)
        col1, col2, col3 = st.columns(3)
        col1.metric("Current kVAR", f"{res['current_kvar']:.0f} kVAR")
        col2.metric("Target kVAR", f"{res['target_kvar']:.0f} kVAR")
        col3.metric("Required Capacitor", f"{res['required_capacitor_kvar']:.0f} kVAR")
        st.info(f"💰 Estimated savings: {res['estimated_savings_pct']:.1f}% reduction in reactive power charges")


def render_containment_tab(calc):
    st.header("📦 Cable Containment Sizing")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Socket Outlets & Switches")
        q13s = st.number_input("13A Switched Socket Outlets", 0, value=0, step=1, key="con_13s")
        q13l = st.number_input("13A Switches (Light)", 0, value=0, step=1, key="con_13l")
        q13u = st.number_input("13A Unswitched Socket Outlets", 0, value=0, step=1, key="con_13u")
        q20 = st.number_input("20A Isolators", 0, value=0, step=1, key="con_20")
        q32 = st.number_input("32A Isolators", 0, value=0, step=1, key="con_32")
    with c2:
        st.subheader("Other Points")
        qlight = st.number_input("Lighting Points", 0, value=0, step=1, key="con_light")
        qfan = st.number_input("Fan Points", 0, value=0, step=1, key="con_fan")
        qwh = st.number_input("Water Heater Points", 0, value=0, step=1, key="con_wh")
        qac = st.number_input("AC Points (13A)", 0, value=0, step=1, key="con_ac")
        qcook = st.number_input("Cooker Points (45A)", 0, value=0, step=1, key="con_cook")
    st.subheader("Custom Cable Entries")
    num_custom = st.number_input("Number of custom cable types", 0, 10, 0, step=1, key="con_custom_num")
    custom = []
    for i in range(num_custom):
        ca, cb, cc = st.columns(3)
        with ca:
            desc = st.text_input(f"Description {i+1}", key=f"con_desc_{i}")
        with cb:
            sz = st.selectbox(f"Cable size {i+1}", list(calc.cable_outer_diameter.keys()), key=f"con_sz_{i}")
        with cc:
            qty = st.number_input(f"Quantity {i+1}", 0, value=0, step=1, key=f"con_qty_{i}")
        if desc and qty > 0:
            custom.append(ContainmentItem(desc, sz, qty, calc.cable_outer_diameter[sz]))
    fill = st.slider("Cable Fill Ratio (typical 40%)", 0.2, 0.6, 0.4, 0.05, key="con_fill")
    if st.button("Calculate Containment Size", type="primary", key="con_btn"):
        items = []
        # Standard mapping
        mapping = [
            ("13A Switched Socket Outlet", q13s, "2.5"),
            ("13A Switch (Light)", q13l, "1.5"),
            ("13A Unswitched Socket", q13u, "2.5"),
            ("20A Isolator", q20, "4"),
            ("32A Isolator", q32, "6"),
            ("Lighting Point", qlight, "1.5"),
            ("Fan Point", qfan, "1.5"),
            ("Water Heater Point", qwh, "4"),
            ("AC Point (13A)", qac, "2.5"),
            ("Cooker Point (45A)", qcook, "10")
        ]
        for desc, qty, sz in mapping:
            if qty > 0:
                items.append(ContainmentItem(desc, sz, qty, calc.cable_outer_diameter[sz]))
        items.extend(custom)
        if not items:
            st.warning("Please enter at least one item.")
            return
        res = calc.calculate_containment(items, fill)
        cola, colb, colc = st.columns(3)
        cola.metric("Total Cable Area", f"{res['total_area_mm2']:.0f} mm²")
        colb.metric("Required Area (with fill)", f"{res['required_area_mm2']:.0f} mm²")
        colc.metric("Recommended Size", res['recommended_size'])
        if res['recommended_size'] != "Exceeds standard sizes, custom required":
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['fill_percentage'],
                domain={'x': [0,1], 'y':[0,1]},
                title={'text': "Fill Percentage"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0,40], 'color': "lightgreen"},
                        {'range': [40,60], 'color': "yellow"},
                        {'range': [60,100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': fill*100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        if res['breakdown']:
            st.subheader("Cable Area Breakdown")
            st.dataframe(pd.DataFrame(res['breakdown']), use_container_width=True)


def render_results_tab(calc, unit_counts, inst_counts, fac_loads, shops, hawk):
    st.header("📊 Electrical Load Summary")
    res_load = calc.calculate_residential_load(unit_counts)
    com_load = calc.calculate_common_load(inst_counts)
    fac_total = 0
    for _, v in fac_loads.items():
        if v:
            fac_total += float(v)
    retail_res = calc.calculate_retail_load(shops, apply_diversity=True)
    unmetered = res_load + com_load + fac_total
    metered = retail_res['demand_kva']
    total = unmetered + metered
    st.subheader("Load Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Residential", f"{res_load:.2f} kVA")
    c1.metric("Common Services", f"{com_load:.2f} kVA")
    c2.metric("Facilities", f"{fac_total:.2f} kVA")
    c2.metric("Retail (Connected)", f"{retail_res['total_kva']:.2f} kVA")
    c3.metric("Retail (Demand)", f"{retail_res['demand_kva']:.2f} kVA")
    c3.metric("Diversity Factor", f"{retail_res['diversity_factor']:.2f}")
    c4.metric("Unmetered Supply", f"{unmetered:.2f} kVA")
    c4.metric("Metered Supply", f"{metered:.2f} kVA")
    c4.metric("**TOTAL LOAD**", f"**{total:.2f} kVA**", delta=f"{total/1000:.2f} MVA")
    # Detailed breakdown for charts
    data = []
    for ut, cnt in unit_counts.items():
        if cnt > 0:
            data.append({"Category": "Residential", "Item": ut, "Load (kVA)": calc.residential_units[ut] * cnt})
    for inst, cnt in inst_counts.items():
        if cnt > 0 and inst in calc.common_installations:
            data.append({"Category": "Common Services", "Item": inst, "Load (kVA)": calc.common_installations[inst] * cnt})
    for fac, l in fac_loads.items():
        if l and l > 0:
            data.append({"Category": "Facilities", "Item": fac, "Load (kVA)": l})
    for s in shops:
        if s:
            data.append({"Category": "Retail", "Item": f"{s.name} ({s.area_sqm:.0f}m²)", "Load (kVA)": s.load_kva})
    if data:
        df = pd.DataFrame(data)
        colp, colb = st.columns(2)
        with colp:
            fig = px.pie(df, values='Load (kVA)', names='Category', title='Load Distribution by Category')
            st.plotly_chart(fig, use_container_width=True)
        with colb:
            top10 = df.sort_values('Load (kVA)', ascending=False).head(10)
            fig2 = px.bar(top10, x='Item', y='Load (kVA)', color='Category', title='Top 10 Individual Loads')
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df.sort_values('Load (kVA)', ascending=False), use_container_width=True, hide_index=True)
    # Transformer sizing
    st.subheader("🔋 Transformer Sizing")
    future = st.slider("Future Expansion Factor (%)", 0, 50, 10, key="tx_future") / 100
    req = total * (1 + future)
    std = [500, 800, 1000, 1250, 1500, 1600, 2000, 2500, 3000]
    if req <= 1000:
        rec = min([s for s in std if s >= req])
        n = 1
        cap = rec
    else:
        n = math.ceil(req / 1000)
        cap = n * 1000
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Required Capacity", f"{req:.0f} kVA")
    tc2.metric("Recommended", f"{n} x 1000 kVA")
    tc3.metric("Margin", f"{cap - req:.0f} kVA ({(cap-req)/req*100:.1f}%)")
    # Generator sizing
    st.subheader("⚡ Standby Generator Sizing")
    emerg = 0
    emerg += fac_loads.get("Polyclinic", 0) * 0.4
    emerg += fac_loads.get("KDC", 0) * 0.3
    emerg += fac_loads.get("CC", 0) * 0.2
    emerg += inst_counts.get("Service Lift (20 man)", 0) * 35 * 0.5
    emerg += inst_counts.get("Fire Hose Reel Pump", 0) * 2.6
    emerg += inst_counts.get("Escalator", 0) * 22 * 0.3
    emerg += inst_counts.get("Public Lighting Circuit", 0) * 12 * 0.8
    gen = math.ceil(emerg / 50) * 50
    gen = max(gen, 200)
    gc1, gc2 = st.columns(2)
    gc1.metric("Emergency Load", f"{emerg:.0f} kVA")
    gc2.metric("Recommended Generator", f"{gen:.0f} kVA")
    largest = max(
        inst_counts.get("Escalator", 0) * 22,
        inst_counts.get("Service Lift (20 man)", 0) * 35,
        inst_counts.get("Fire Hose Reel Pump", 0) * 2.6
    )
    if largest * 3 > gen:
        st.warning(f"⚠️ Generator may need upsizing for motor starting (largest motor: {largest:.0f} kVA)")
    # Professional declaration
    st.subheader("📝 Professional Engineer's Declaration")
    st.markdown("""
    I, the Professional Engineer for the declared Electrical Works, hereby submit the electrical design load data and calculation and confirm that the details given in this form are to the best of my knowledge true and correct.
    """)
    pe1, pe2 = st.columns(2)
    pe1.text_input("PE Name", "Ting Ik Hing", key="pe_name_res")
    pe1.text_input("Registration No.", "3348", key="pe_reg_res")
    pe2.text_input("Firm Name", "Surbana International Consultants Pte Ltd", key="firm_res")
    pe2.text_input("Date", datetime.now().strftime("%Y-%m-%d"), key="date_res")


def render_export_tab(calc, unit_counts, inst_counts, fac_loads, shops, hawk):
    st.header("📥 Export Options")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📄 Export to Excel", use_container_width=True):
        # In a real app you would generate an Excel file
        st.success("Excel export simulated")
    if c2.button("📊 Export to PDF", use_container_width=True):
        st.info("PDF export simulated")
    if c3.button("💾 Save Project", use_container_width=True):
        st.session_state['saved'] = True
        st.success("Project saved to session")
    if c4.button("📋 Copy Summary", use_container_width=True):
        st.success("Summary copied (simulated)")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="HDB Electrical Load Calculator", page_icon="⚡", layout="wide")
    st.markdown("""
    <style>
    .stApp { background-color: #f5f7f9; }
    .main-header {
        background-color: #003366; padding: 1rem; border-radius: 0.5rem;
        color: white; text-align: center; margin-bottom: 2rem;
    }
    .stMetric { background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header"><h1>⚡ HDB Electrical Design Load Calculator</h1><p>Professional Edition - With W/m² Retail & Containment Sizing</p></div>',
                unsafe_allow_html=True)

    calc = ElectricalLoadCalculator()
    proj_info = render_sidebar()

    tabs = st.tabs(["🏢 Residential", "⚙️ Common Services", "🏛️ Facilities", "🛍️ Retail",
                    "🍜 Hawker Centre", "🔌 Distribution", "📦 Containment", "📊 Results"])

    with tabs[0]:
        unit_counts = render_residential_tab(calc)
    with tabs[1]:
        inst_counts = render_common_tab(calc)
    with tabs[2]:
        fac_loads = render_facilities_tab(calc)
    with tabs[3]:
        shops = render_retail_tab(calc)
    with tabs[4]:
        hawk = render_hawker_tab()
    with tabs[5]:
        sub_tabs = st.tabs(["Distribution System", "Voltage Drop", "Fault Current"])
        with sub_tabs[0]:
            render_distribution_tab()
        with sub_tabs[1]:
            render_voltage_drop_tab(calc)
        with sub_tabs[2]:
            render_fault_current_tab(calc)
    with tabs[6]:
        render_containment_tab(calc)
    with tabs[7]:
        render_results_tab(calc, unit_counts, inst_counts, fac_loads, shops, hawk)
        render_export_tab(calc, unit_counts, inst_counts, fac_loads, shops, hawk)
        with st.expander("Power Factor Correction Calculator"):
            render_pfc_tab(calc)

    st.divider()
    st.caption(f"© 2024 HDB - Electrical Design Load Calculator v4.0 | Project: {proj_info['project_title']} | Reference: {proj_info['project_ref']}")


if __name__ == "__main__":
    main()
