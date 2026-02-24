# app.py - Complete final version with all fixes

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import math
import io

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ElectricalLoad:
    """Base class for electrical loads"""
    name: str
    load_kva: float
    diversity_factor: float = 1.0
    
    @property
    def demand_load(self) -> float:
        return self.load_kva * self.diversity_factor

@dataclass
class ResidentialUnit:
    unit_type: str
    load_kva: float
    quantity: int = 0
    
    @property
    def total_load(self) -> float:
        return self.load_kva * self.quantity

@dataclass
class RetailShop:
    name: str
    area_sqm: float
    load_density_w_per_sqm: float  # W/m²
    breaker_size: str
    power_factor: float = 0.85
    
    @property
    def load_kw(self) -> float:
        """Calculate load in kW based on area and load density"""
        return (self.area_sqm * self.load_density_w_per_sqm) / 1000
    
    @property
    def load_kva(self) -> float:
        return self.load_kw / self.power_factor
    
    def __post_init__(self):
        """Validate the data after initialization"""
        if self.area_sqm < 0:
            raise ValueError("Area cannot be negative")
        if self.load_density_w_per_sqm < 0:
            raise ValueError("Load density cannot be negative")

@dataclass
class CableSizing:
    """Cable sizing parameters"""
    current_a: float
    length_m: float
    voltage_drop_percent: float
    cable_size_mm2: str
    cable_type: str
    
@dataclass
class VoltageDrop:
    """Voltage drop calculation results"""
    percentage: float
    is_compliant: bool
    recommended_cable: str

@dataclass
class ContainmentItem:
    """Represents an item contributing to cable containment"""
    description: str
    cable_size_mm2: str      # e.g., "2.5"
    quantity: int
    cable_outer_diameter_mm: float   # overall diameter including insulation

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class ElectricalLoadCalculator:
    """Main calculator class for electrical loads"""
    
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
        
        # Load densities for different retail types (W/m²)
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
            "Chinese Medicine": 168,  # Based on Excel data
            "Halal Café": 228,  # Based on Excel data (27.712kW/121.34m²)
            "Takeaway Food": 285,  # Based on Excel data
        }
        
        # Diversity factors by load type
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
        
        # Cable reference data (mm² -> current capacity A)
        self.cable_current_capacity = {
            "1.5": 17.5,
            "2.5": 24,
            "4": 32,
            "6": 41,
            "10": 57,
            "16": 76,
            "25": 101,
            "35": 125,
            "50": 151,
            "70": 192,
            "95": 232,
            "120": 269,
            "150": 300,
            "185": 341,
            "240": 400,
            "300": 458
        }
        
        # Cable outer diameters (including insulation) in mm for typical PVC/PVC cables
        self.cable_outer_diameter = {
            "1.5": 8.0,
            "2.5": 8.8,
            "4": 9.8,
            "6": 10.9,
            "10": 12.8,
            "16": 14.9,
            "25": 18.2,
            "35": 20.5,
            "50": 23.4,
            "70": 26.5,
            "95": 29.8,
            "120": 32.6,
            "150": 35.8,
            "185": 39.2,
            "240": 44.5,
            "300": 49.0
        }
        
        # Standard containment sizes (width x height in mm) - tray/trunking
        self.standard_containment_sizes = [
            (50, 50), (75, 50), (100, 50), (100, 75), (100, 100),
            (150, 50), (150, 75), (150, 100), (150, 150),
            (200, 50), (200, 75), (200, 100), (200, 150), (200, 200),
            (300, 100), (300, 150), (300, 200), (300, 300),
            (400, 100), (400, 150), (400, 200), (400, 300), (400, 400),
            (500, 100), (500, 150), (500, 200), (500, 300), (500, 400), (500, 500)
        ]
        
        # Mapping from accessory type to typical cable size
        self.accessory_cable_map = {
            "13A Switched Socket Outlet": "2.5",
            "13A Switch (Light)": "1.5",
            "20A Isolator": "4",
            "32A Isolator": "6",
            "Lighting Point": "1.5",
            "13A Unswitched Socket": "2.5",
            "Fan Point": "1.5",
            "Water Heater Point": "4",
            "Air Conditioner Point (13A)": "2.5",
            "Air Conditioner Point (15A)": "2.5",
            "Cooker Point (45A)": "10"
        }
        
    def calculate_residential_load(self, unit_counts: Dict[str, int]) -> float:
        """Calculate total residential load"""
        total = 0
        for unit_type, count in unit_counts.items():
            if unit_type in self.residential_units and count > 0:
                total += self.residential_units[unit_type] * count
        return total
    
    def calculate_common_load(self, installation_counts: Dict[str, int]) -> float:
        """Calculate common services load"""
        total = 0
        for install, count in installation_counts.items():
            if install in self.common_installations and count > 0:
                total += self.common_installations[install] * count
        return total
    
    def calculate_retail_load(self, shops: List[RetailShop], apply_diversity: bool = True) -> Dict:
        """Calculate retail shops load with optional diversity"""
        if not shops:
            return {
                "total_kw": 0,
                "total_kva": 0,
                "diversity_factor": 1.0,
                "demand_kw": 0,
                "demand_kva": 0,
                "shops": shops
            }
        
        total_kw = sum(shop.load_kw for shop in shops if shop and hasattr(shop, 'load_kw'))
        total_kva = sum(shop.load_kva for shop in shops if shop and hasattr(shop, 'load_kva'))
        
        diversity = self.diversity_factors.get("retail", 0.85) if apply_diversity else 1.0
        
        return {
            "total_kw": total_kw,
            "total_kva": total_kva,
            "diversity_factor": diversity,
            "demand_kw": total_kw * diversity,
            "demand_kva": total_kva * diversity,
            "shops": shops
        }
    
    def calculate_hawker_centre(self, area_sqm: float, cooked_food_stalls: int = 40, 
                                lighting_density: float = 20, mv_motors: int = 8,
                                mv_motor_power: float = 11, office_ac: float = 3) -> Dict:
        """Calculate hawker centre load based on typical configuration"""
        cooked_food_load = cooked_food_stalls * 8  # 8kW per stall
        lighting_load = area_sqm * lighting_density / 1000  # Convert W/m² to kW
        mech_vent_load = mv_motors * mv_motor_power
        
        total_kw = cooked_food_load + lighting_load + mech_vent_load + office_ac
        total_kva = total_kw / 0.85
        
        diversity = self.diversity_factors.get("hawker", 0.8)
        
        return {
            "cooked_food_stalls": cooked_food_stalls,
            "cooked_food_load_kw": cooked_food_load,
            "lighting_load_kw": lighting_load,
            "mech_vent_load_kw": mech_vent_load,
            "office_ac_kw": office_ac,
            "total_kw": total_kw,
            "total_kva": total_kva,
            "diversity_factor": diversity,
            "demand_kw": total_kw * diversity,
            "demand_kva": total_kva * diversity
        }
    
    def calculate_polyclinic(self, gfa_sqm: float) -> Dict:
        """Calculate polyclinic load"""
        normal_supply = gfa_sqm * 0.120  # 120W/m²
        emergency_supply = gfa_sqm * 0.055  # 55W/m²
        
        return {
            "gfa_sqm": gfa_sqm,
            "normal_supply_kw": normal_supply,
            "normal_supply_kva": normal_supply / 0.85,
            "emergency_supply_kw": emergency_supply,
            "emergency_supply_kva": emergency_supply / 0.85,
            "total_kva": (normal_supply + emergency_supply) / 0.85
        }
    
    def calculate_voltage_drop(self, current_a: float, length_m: float, 
                               cable_size_mm2: str, power_factor: float = 0.85,
                               voltage_v: int = 400) -> VoltageDrop:
        """Calculate voltage drop for a cable"""
        # Resistance and reactance per km (typical values for copper)
        cable_data = {
            "1.5": {"r": 14.8, "x": 0.155},
            "2.5": {"r": 8.91, "x": 0.145},
            "4": {"r": 5.57, "x": 0.135},
            "6": {"r": 3.71, "x": 0.13},
            "10": {"r": 2.24, "x": 0.125},
            "16": {"r": 1.41, "x": 0.12},
            "25": {"r": 0.89, "x": 0.115},
            "35": {"r": 0.67, "x": 0.11},
            "50": {"r": 0.49, "x": 0.105},
            "70": {"r": 0.35, "x": 0.1},
            "95": {"r": 0.26, "x": 0.095},
            "120": {"r": 0.21, "x": 0.09},
            "150": {"r": 0.17, "x": 0.085},
            "185": {"r": 0.14, "x": 0.08},
            "240": {"r": 0.11, "x": 0.075},
            "300": {"r": 0.09, "x": 0.07}
        }
        
        if cable_size_mm2 not in cable_data:
            return VoltageDrop(percentage=999, is_compliant=False, recommended_cable="Unknown")
        
        data = cable_data[cable_size_mm2]
        
        # Voltage drop calculation: Vd = √3 × I × L × (Rcosφ + Xsinφ) / 1000
        sin_phi = math.sqrt(1 - power_factor**2)
        
        voltage_drop_v = math.sqrt(3) * current_a * (length_m/1000) * \
                        (data["r"] * power_factor + data["x"] * sin_phi)
        
        voltage_drop_percent = (voltage_drop_v / voltage_v) * 100
        
        # Check compliance (typically 4% for lighting, 6% for power)
        is_compliant = voltage_drop_percent <= 4.0
        
        # Recommend larger cable if not compliant
        recommended = cable_size_mm2
        if not is_compliant:
            for size in sorted([float(s) for s in cable_data.keys()]):
                if float(size) > float(cable_size_mm2):
                    vd_test = self.calculate_voltage_drop(current_a, length_m, str(int(size)), power_factor, voltage_v)
                    if vd_test.percentage <= 4.0:
                        recommended = str(int(size))
                        break
        
        return VoltageDrop(
            percentage=voltage_drop_percent,
            is_compliant=is_compliant,
            recommended_cable=recommended
        )
    
    def calculate_fault_current(self, transformer_kva: float, impedance_pct: float = 5.0,
                                voltage_v: int = 400) -> Dict:
        """Calculate prospective fault current"""
        # Full load current
        flc = transformer_kva * 1000 / (math.sqrt(3) * voltage_v)
        
        # Fault current (assuming infinite bus)
        fault_current = (flc * 100) / impedance_pct
        
        # Peak current (for breaker sizing)
        peak_current = fault_current * 2.5  # Typical factor for peak making capacity
        
        return {
            "transformer_kva": transformer_kva,
            "full_load_current_a": flc,
            "fault_current_ka": fault_current / 1000,
            "peak_current_ka": peak_current / 1000,
            "impedance_pct": impedance_pct
        }
    
    def calculate_power_factor_correction(self, load_kw: float, current_pf: float, 
                                          target_pf: float = 0.95) -> Dict:
        """Calculate required capacitor bank for power factor correction"""
        # Calculate reactive power at current PF
        current_kvar = load_kw * math.tan(math.acos(current_pf))
        
        # Calculate reactive power at target PF
        target_kvar = load_kw * math.tan(math.acos(target_pf))
        
        # Required capacitor kvar
        required_kvar = current_kvar - target_kvar
        
        return {
            "load_kw": load_kw,
            "current_pf": current_pf,
            "target_pf": target_pf,
            "current_kvar": current_kvar,
            "target_kvar": target_kvar,
            "required_capacitor_kvar": required_kvar,
            "estimated_savings_pct": ((current_pf - target_pf) / current_pf) * 100
        }
    
    def calculate_containment_size(self, items: List[ContainmentItem], fill_ratio: float = 0.4) -> Dict:
        """Calculate required containment size based on items and fill ratio"""
        if not items:
            return {"total_area_mm2": 0, "recommended_size": "None", "fill_percentage": 0}
        
        # Calculate total cross-sectional area of cables (assuming circular)
        total_area = 0
        breakdown = []
        for item in items:
            if item.quantity > 0 and item.cable_size_mm2 in self.cable_outer_diameter:
                dia = self.cable_outer_diameter[item.cable_size_mm2]
                area_per_cable = math.pi * (dia/2)**2  # mm²
                item_area = area_per_cable * item.quantity
                total_area += item_area
                breakdown.append({
                    "Description": item.description,
                    "Cable Size (mm²)": item.cable_size_mm2,
                    "Quantity": item.quantity,
                    "Area per Cable (mm²)": round(area_per_cable, 2),
                    "Total Area (mm²)": round(item_area, 2)
                })
        
        # Required containment area considering fill ratio
        required_area = total_area / fill_ratio
        
        # Find smallest standard containment that meets required_area
        recommended_size = None
        for width, height in sorted(self.standard_containment_sizes, key=lambda x: x[0]*x[1]):
            area = width * height
            if area >= required_area:
                recommended_size = f"{width} x {height} mm"
                break
        
        if recommended_size is None:
            recommended_size = "Exceeds standard sizes, custom required"
        
        # Calculate actual fill percentage if we pick that size
        if recommended_size != "Exceeds standard sizes, custom required":
            # Find dimensions
            w, h = map(int, recommended_size.replace(' mm','').split(' x '))
            containment_area = w * h
            fill_pct = (total_area / containment_area) * 100
        else:
            fill_pct = 0
        
        return {
            "total_area_mm2": total_area,
            "required_area_mm2": required_area,
            "fill_ratio": fill_ratio,
            "recommended_size": recommended_size,
            "fill_percentage": fill_pct,
            "breakdown": breakdown
        }

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render sidebar with project info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100.png?text=HDB+Logo", use_column_width=True)
        st.title("Project Information")
        
        project_title = st.text_input("Project Title", "PROPOSED PUBLIC HOUSING DEVELOPMENT")
        project_ref = st.text_input("Project Reference No.", "Axxx")
        location = st.text_input("Location Description", "CCK")
        
        st.divider()
        
        st.subheader("Professional Engineer")
        pe_name = st.text_input("Name", "Ting Ik Hing")
        pe_reg_no = st.text_input("Registration No.", "3348")
        firm_name = st.text_input("Firm Name", "Surbana International Consultants Pte Ltd")
        telephone = st.text_input("Telephone No.", "62481315")
        
        st.divider()
        
        st.date_input("Date", datetime.now())
        
        return {
            "project_title": project_title,
            "project_ref": project_ref,
            "location": location,
            "pe_name": pe_name,
            "pe_reg_no": pe_reg_no,
            "firm_name": firm_name,
            "telephone": telephone
        }

def render_residential_tab(calculator):
    """Render residential units input tab (fixed sample button)"""
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

def render_common_services_tab(calculator):
    """Render common services input tab"""
    st.header("⚙️ Common Services")
    
    installation_counts = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lifts & Escalators")
        installation_counts["Escalator"] = st.number_input("Escalator (22 kVA)", 0, 10, 2, key="common_esc")
        installation_counts["Service Lift (20 man)"] = st.number_input("Service Lift (20 man, 35 kVA)", 0, 10, 1, key="common_sl")
        installation_counts["Lift 11 sty (13 man)"] = st.number_input("Lift 11 sty (13 man, 20 kVA)", 0, 10, 1, key="common_lift11")
        installation_counts["Lift (20 man)"] = st.number_input("Lift (20 man, 35 kVA)", 0, 10, 0, key="common_lift20")
    
    with col2:
        st.subheader("Pumps & Lighting")
        installation_counts["Domestic Booster Pump"] = st.number_input("Domestic Booster Pump (3.52 kVA)", 0, 10, 0, key="common_dbp")
        installation_counts["Fire Hose Reel Pump"] = st.number_input("Fire Hose Reel Pump (2.6 kVA)", 0, 10, 1, key="common_fhr")
        installation_counts["Public Lighting Circuit"] = st.number_input("Public Lighting Circuit (12 kVA)", 0, 10, 1, key="common_pl")
        installation_counts["Outdoor Lighting Circuit"] = st.number_input("Outdoor Lighting Circuit (17 kVA)", 0, 10, 1, key="common_ol")
    
    st.subheader("Fire Protection Systems")
    col3, col4 = st.columns(2)
    with col3:
        installation_counts["Sprinkler System"] = st.number_input("Sprinkler System (44 kVA)", 0, 10, 0, key="common_spr")
        installation_counts["Wet Riser System"] = st.number_input("Wet Riser System (25 kVA)", 0, 10, 0, key="common_wr")
    with col4:
        installation_counts["Refuse Handling Plant"] = st.number_input("Refuse Handling Plant (10 kVA)", 0, 10, 1, key="common_rh")
        installation_counts["Mech Ventilation"] = st.number_input("Mech Ventilation (11 kVA)", 0, 20, 0, key="common_mv")
    
    return installation_counts

def render_facilities_tab(calculator):
    """Render facilities input tab"""
    st.header("🏛️ Facilities")
    
    col1, col2 = st.columns(2)
    
    facility_loads = {}
    
    with col1:
        st.subheader("Community Facilities")
        facility_loads["Future Communal Facilities"] = st.number_input("Future Communal Facilities (42 kVA)", 0, 1000, 0, key="fac_fcf")
        facility_loads["Bin Centre"] = st.number_input("Bin Centre (10 kVA)", 0, 100, 0, key="fac_bin")
        facility_loads["RC Centre"] = st.number_input("RC Centre (42 kVA)", 0, 100, 0, key="fac_rc")
        facility_loads["CC"] = st.number_input("CC (693 kVA)", 0, 2000, 0, key="fac_cc")
    
    with col2:
        st.subheader("Healthcare")
        facility_loads["Polyclinic"] = st.number_input("Polyclinic (881.28 kVA)", 0, 2000, 0, key="fac_poly")
        facility_loads["KDC"] = st.number_input("KDC (104 kVA)", 0, 500, 0, key="fac_kdc")
        
        st.subheader("Others")
        facility_loads["Hawker Centre"] = st.number_input("Hawker Centre (676.88 kVA/3500sqm)", 0, 2000, 0, key="fac_hawker")
        facility_loads["MNO"] = st.number_input("MNO (41.6 kVA)", 0, 500, 0, key="fac_mno")
        facility_loads["EPS"] = st.number_input("EPS (14 kVA)", 0, 500, 0, key="fac_eps")
        facility_loads["Mech"] = st.number_input("Mech (762 kVA)", 0, 2000, 0, key="fac_mech")
    
    return facility_loads

def render_retail_shops(calculator):
    """Render retail shops input with W/m² calculation"""
    st.header("🛍️ Retail Shops Details")
    
    st.info("Load is calculated based on Area (m²) × Load Density (W/m²)")
    
    # Sample shops from Excel with calculated load densities
    sample_shops = [
        {"name": "Retail (Chinese medicine)", "area": 86.3, "load_density": 168, "breaker": "63A SPN"},
        {"name": "Convenience Store", "area": 121.28, "load_density": 228, "breaker": "40A TPN"},
        {"name": "Open Retail", "area": 27.98, "load_density": 518, "breaker": "63A SPN"},
        {"name": "Halal Café", "area": 121.34, "load_density": 228, "breaker": "40A TPN"},
        {"name": "Takeaway food", "area": 50.85, "load_density": 285, "breaker": "63A SPN"},
        {"name": "Gym", "area": 299.93, "load_density": 185, "breaker": "80A TPN"}
    ]
    
    shops = []
    
    # Initialize session state for shops if not exists
    if 'shops_w_per_sqm' not in st.session_state:
        st.session_state.shops_w_per_sqm = sample_shops.copy()
    
    # Create dataframe for editing
    df_shops = pd.DataFrame(st.session_state.shops_w_per_sqm)
    
    # Calculate load_kw for display
    df_shops['load_kw'] = (df_shops['area'] * df_shops['load_density']) / 1000
    
    # Prepare display columns
    display_cols = ['name', 'area', 'load_density', 'breaker', 'load_kw']
    display_df = df_shops[display_cols].rename(columns={
        'name': 'Shop Name',
        'area': 'Area (m²)',
        'load_density': 'Load Density (W/m²)',
        'breaker': 'Breaker Size',
        'load_kw': 'Load (kW)'
    })
    
    # Create column config
    column_config = {
        "Shop Name": st.column_config.TextColumn("Shop Name", required=True),
        "Area (m²)": st.column_config.NumberColumn("Area (m²)", min_value=0, format="%.2f", required=True),
        "Load Density (W/m²)": st.column_config.NumberColumn("Load Density (W/m²)", min_value=0, format="%.0f", required=True),
        "Breaker Size": st.column_config.TextColumn("Breaker Size", required=True),
        "Load (kW)": st.column_config.NumberColumn("Load (kW)", format="%.3f", disabled=True)
    }
    
    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config=column_config
    )
    
    # Convert back to original format
    if not edited_df.empty:
        updated_shops = []
        for _, row in edited_df.iterrows():
            shop_dict = {
                "name": row["Shop Name"],
                "area": row["Area (m²)"],
                "load_density": row["Load Density (W/m²)"],
                "breaker": row["Breaker Size"]
            }
            updated_shops.append(shop_dict)
        
        st.session_state.shops_w_per_sqm = updated_shops
        
        # Display calculated loads
        st.subheader("Calculated Shop Loads")
        calc_df = pd.DataFrame(updated_shops)
        calc_df['Load (kW)'] = (calc_df['area'] * calc_df['load_density']) / 1000
        calc_df['Load (kVA)'] = calc_df['Load (kW)'] / 0.85
        calc_df['Current (A)'] = calc_df['Load (kW)'] * 1000 / (1.732 * 400 * 0.85)
        
        st.dataframe(
            calc_df[['name', 'area', 'load_density', 'Load (kW)', 'Load (kVA)', 'Current (A)']].rename(columns={
                'name': 'Shop Name',
                'area': 'Area (m²)',
                'load_density': 'W/m²',
                'Load (kW)': 'Load (kW)',
                'Load (kVA)': 'Load (kVA)',
                'Current (A)': 'Current (A)'
            }).round(2),
            use_container_width=True
        )
        
        # Convert to RetailShop objects
        try:
            shops = [RetailShop(
                name=shop["name"],
                area_sqm=shop["area"],
                load_density_w_per_sqm=shop["load_density"],
                breaker_size=shop["breaker"],
                power_factor=0.85
            ) for shop in st.session_state.shops_w_per_sqm]
        except Exception as e:
            st.error(f"Error creating shop objects: {str(e)}")
            shops = []
    
    return shops

def render_hawker_centre_detail():
    """Render detailed hawker centre calculation"""
    st.header("🍜 Hawker Centre Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sqm)", 0, 10000, 3500, key="hawker_area")
        cooked_food_stalls = st.number_input("Number of Cooked Food Stalls", 0, 100, 40, key="hawker_stalls")
        lighting_density = st.number_input("Lighting Density (W/m²)", 0.0, 50.0, 20.0, key="hawker_light")
    
    with col2:
        mv_motors = st.number_input("Number of MV Motors", 0, 20, 8, key="hawker_mv")
        mv_motor_power = st.number_input("MV Motor Power (kW)", 0.0, 50.0, 11.0, key="hawker_mv_power")
        management_office_ac = st.number_input("Management Office AC (kW)", 0.0, 20.0, 3.0, key="hawker_ac")
        diversity_factor = st.slider("Diversity Factor", 0.5, 1.0, 0.8, 0.05, key="hawker_div")
    
    # Calculate loads
    cooked_food_load = cooked_food_stalls * 8
    lighting_load = area * lighting_density / 1000
    mv_load = mv_motors * mv_motor_power
    total_kw = cooked_food_load + lighting_load + mv_load + management_office_ac
    total_kva = total_kw / 0.85
    
    # Display results
    st.subheader("Hawker Centre Load Summary")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Connected Load", f"{total_kw:.2f} kW")
    with col4:
        st.metric("Connected Load", f"{total_kva:.2f} kVA")
    with col5:
        st.metric("Demand Load (with diversity)", f"{total_kva * diversity_factor:.2f} kVA")
    
    results_df = pd.DataFrame({
        "Description": ["Cooked Food Stalls", "Lighting", "MV Motors", "Management Office AC", "TOTAL"],
        "Load (kW)": [cooked_food_load, lighting_load, mv_load, management_office_ac, total_kw],
        "Load (kVA)": [cooked_food_load/0.85, lighting_load/0.85, mv_load/0.85, management_office_ac/0.85, total_kva]
    })
    st.dataframe(results_df, use_container_width=True)
    
    return {
        "area": area,
        "cooked_food_stalls": cooked_food_stalls,
        "total_kw": total_kw,
        "total_kva": total_kva,
        "demand_kva": total_kva * diversity_factor
    }

def render_distribution_system():
    """Render distribution system diagram"""
    st.header("🔌 Distribution System")
    
    # Distribution data from Excel
    distribution_data = {
        "PG Incoming": ["PG Incoming 1", "PG Incoming 2", "PG Incoming 3", "PG Incoming 4"],
        "MSB": ["MSB 1", "MSB 1", "MSB 2", "MSB 2"],
        "SSB": ["SSB-Resi", "SSB-PC", "SSB-CC", "SSB-HWC"],
        "Other": ["SSB-Shop, Chiller Plant", "SSB-KDC", "EMSB-1", "EMSB-2"],
        "Load (kVA)": [1067, 985, 951, 925]
    }
    
    df_dist = pd.DataFrame(distribution_data)
    st.dataframe(df_dist, use_container_width=True)
    
    # Create a more detailed distribution diagram
    fig = go.Figure()
    
    # Add main nodes
    fig.add_trace(go.Scatter(
        x=[0, 2, 4, 6, 8],
        y=[5, 5, 5, 5, 5],
        mode='markers+text',
        marker=dict(size=[40, 30, 30, 30, 30], color=['darkblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']),
        text=['Source', 'PG1', 'PG2', 'PG3', 'PG4'],
        textposition="middle center",
        textfont=dict(color='white', size=12),
        name=''
    ))
    
    # Add load nodes
    loads = ['Resi (1067)', 'PC (985)', 'CC (951)', 'HWC (925)']
    for i, load in enumerate(loads):
        fig.add_trace(go.Scatter(
            x=[2 + i*2, 2 + i*2],
            y=[4, 3],
            mode='lines+markers+text',
            line=dict(color='gray', width=2),
            marker=dict(size=20, color='orange'),
            text=[f'', load],
            textposition="bottom center",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Single Line Diagram",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1, 10]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[2, 6]),
        height=400,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_voltage_drop_calculator(calculator):
    """Render voltage drop calculator tool"""
    st.header("⚡ Voltage Drop Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current = st.number_input("Current (A)", min_value=0.0, value=100.0, step=10.0, key="vd_current")
        length = st.number_input("Cable Length (m)", min_value=0.0, value=50.0, step=5.0, key="vd_length")
        voltage = st.selectbox("System Voltage (V)", [230, 400, 690], index=1, key="vd_voltage")
    
    with col2:
        cable_sizes = list(calculator.cable_current_capacity.keys())
        cable_size = st.selectbox("Cable Size (mm²)", cable_sizes, index=10, key="vd_cable")
        power_factor = st.slider("Power Factor", 0.7, 1.0, 0.85, 0.01, key="vd_pf")
    
    if st.button("Calculate Voltage Drop", key="vd_calc"):
        vd_result = calculator.calculate_voltage_drop(
            current_a=current,
            length_m=length,
            cable_size_mm2=cable_size,
            power_factor=power_factor,
            voltage_v=voltage
        )
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Voltage Drop", f"{vd_result.percentage:.2f}%")
        with col4:
            status = "✅ Compliant" if vd_result.is_compliant else "❌ Not Compliant"
            st.metric("Status", status)
        with col5:
            if not vd_result.is_compliant:
                st.metric("Recommended Cable", f"{vd_result.recommended_cable} mm²")
        
        # Check current capacity
        current_capacity = calculator.cable_current_capacity.get(cable_size, 0)
        if current > current_capacity:
            st.error(f"⚠️ Current ({current}A) exceeds cable capacity ({current_capacity}A)")

def render_fault_current_calculator(calculator):
    """Render fault current calculator"""
    st.header("⚡ Fault Current Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transformer_kva = st.number_input("Transformer Rating (kVA)", min_value=100, value=1000, step=100, key="fc_tx")
        impedance = st.number_input("Transformer Impedance (%)", min_value=1.0, value=5.0, step=0.5, format="%.1f", key="fc_imp")
    
    with col2:
        voltage = st.selectbox("Secondary Voltage (V)", [400, 690], index=0, key="fc_voltage")
    
    if st.button("Calculate Fault Current", key="fc_calc"):
        fault = calculator.calculate_fault_current(transformer_kva, impedance, voltage)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Full Load Current", f"{fault['full_load_current_a']:.0f} A")
        with col4:
            st.metric("Fault Current", f"{fault['fault_current_ka']:.2f} kA")
        with col5:
            st.metric("Peak Current", f"{fault['peak_current_ka']:.2f} kA")
        
        # Breaker recommendations
        st.subheader("Recommended Breaker Ratings")
        breaker_data = pd.DataFrame({
            "Breaker Type": ["MCCB", "ACB"],
            "Rated Current (A)": [fault['full_load_current_a'] * 1.25, fault['full_load_current_a'] * 1.25],
            "Breaking Capacity (kA)": [fault['fault_current_ka'] * 1.1, fault['fault_current_ka'] * 1.1]
        })
        st.dataframe(breaker_data, use_container_width=True)

def render_power_factor_correction(calculator):
    """Render power factor correction calculator"""
    st.header("⚡ Power Factor Correction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        load_kw = st.number_input("Load (kW)", min_value=0.0, value=500.0, step=50.0, key="pfc_load")
        current_pf = st.slider("Current Power Factor", 0.5, 0.95, 0.8, 0.01, key="pfc_cur")
    
    with col2:
        target_pf = st.slider("Target Power Factor", 0.85, 1.0, 0.95, 0.01, key="pfc_tar")
    
    if st.button("Calculate Capacitor Bank", key="pfc_calc"):
        pfc = calculator.calculate_power_factor_correction(load_kw, current_pf, target_pf)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Current kVAR", f"{pfc['current_kvar']:.0f} kVAR")
        with col4:
            st.metric("Target kVAR", f"{pfc['target_kvar']:.0f} kVAR")
        with col5:
            st.metric("Required Capacitor", f"{pfc['required_capacitor_kvar']:.0f} kVAR")
        
        st.info(f"💰 Estimated savings: {pfc['estimated_savings_pct']:.1f}% reduction in reactive power charges")

def render_containment_tab(calculator):
    """Render cable containment sizing tab"""
    st.header("📦 Cable Containment Sizing")
    
    st.markdown("""
    This tool helps you determine the required cable tray or trunking size based on the number of final circuits/accessories.
    Each accessory type is mapped to a typical cable size. The total cross-sectional area of cables is calculated, and a containment size is recommended based on a standard fill ratio.
    """)
    
    # Input quantities for different accessories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Socket Outlets & Switches")
        qty_13a_switched_socket = st.number_input("13A Switched Socket Outlets", min_value=0, value=0, step=1, key="cont_13as")
        qty_13a_switch = st.number_input("13A Switches (Light)", min_value=0, value=0, step=1, key="cont_13al")
        qty_13a_unswitched = st.number_input("13A Unswitched Socket Outlets", min_value=0, value=0, step=1, key="cont_13au")
        qty_20a_isolator = st.number_input("20A Isolators", min_value=0, value=0, step=1, key="cont_20a")
        qty_32a_isolator = st.number_input("32A Isolators", min_value=0, value=0, step=1, key="cont_32a")
    
    with col2:
        st.subheader("Other Points")
        qty_lighting = st.number_input("Lighting Points", min_value=0, value=0, step=1, key="cont_light")
        qty_fan = st.number_input("Fan Points", min_value=0, value=0, step=1, key="cont_fan")
        qty_water_heater = st.number_input("Water Heater Points", min_value=0, value=0, step=1, key="cont_wh")
        qty_ac_13a = st.number_input("AC Points (13A)", min_value=0, value=0, step=1, key="cont_ac")
        qty_cooker = st.number_input("Cooker Points (45A)", min_value=0, value=0, step=1, key="cont_cooker")
    
    # Allow custom cable entries
    st.subheader("Custom Cable Entries (if any)")
    st.info("Add custom cables if your installation includes non-standard circuits.")
    
    custom_items = []
    num_custom = st.number_input("Number of custom cable types", min_value=0, max_value=10, value=0, step=1, key="cont_num_custom")
    
    if num_custom > 0:
        for i in range(num_custom):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                desc = st.text_input(f"Description {i+1}", key=f"custom_desc_{i}")
            with col_b:
                cable_size = st.selectbox(f"Cable size (mm²) {i+1}", 
                                           options=list(calculator.cable_outer_diameter.keys()),
                                           key=f"custom_size_{i}")
            with col_c:
                qty = st.number_input(f"Quantity {i+1}", min_value=0, value=0, step=1, key=f"custom_qty_{i}")
            if desc and qty > 0:
                custom_items.append(ContainmentItem(
                    description=desc,
                    cable_size_mm2=cable_size,
                    quantity=qty,
                    cable_outer_diameter_mm=calculator.cable_outer_diameter[cable_size]
                ))
    
    # Fill ratio
    fill_ratio = st.slider("Cable Fill Ratio (typical 40%)", 0.2, 0.6, 0.4, 0.05, key="cont_fill")
    
    if st.button("Calculate Containment Size", type="primary", key="cont_calc"):
        # Build list of items
        items = []
        
        # Map quantities to ContainmentItem
        mapping = [
            ("13A Switched Socket Outlet", qty_13a_switched_socket, "2.5"),
            ("13A Switch (Light)", qty_13a_switch, "1.5"),
            ("13A Unswitched Socket", qty_13a_unswitched, "2.5"),
            ("20A Isolator", qty_20a_isolator, "4"),
            ("32A Isolator", qty_32a_isolator, "6"),
            ("Lighting Point", qty_lighting, "1.5"),
            ("Fan Point", qty_fan, "1.5"),
            ("Water Heater Point", qty_water_heater, "4"),
            ("AC Point (13A)", qty_ac_13a, "2.5"),
            ("Cooker Point (45A)", qty_cooker, "10")
        ]
        
        for desc, qty, cable_size in mapping:
            if qty > 0:
                items.append(ContainmentItem(
                    description=desc,
                    cable_size_mm2=cable_size,
                    quantity=qty,
                    cable_outer_diameter_mm=calculator.cable_outer_diameter[cable_size]
                ))
        
        # Add custom items
        items.extend(custom_items)
        
        if not items:
            st.warning("Please enter at least one item.")
            return
        
        result = calculator.calculate_containment_size(items, fill_ratio)
        
        # Display results
        st.subheader("Recommended Containment Size")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Total Cable Area", f"{result['total_area_mm2']:.0f} mm²")
        with col_res2:
            st.metric("Required Area (with fill)", f"{result['required_area_mm2']:.0f} mm²")
        with col_res3:
            st.metric("Recommended Size", result['recommended_size'])
        
        # Fill percentage gauge
        if result['recommended_size'] != "Exceeds standard sizes, custom required":
            fill_pct = result['fill_percentage']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fill_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fill Percentage"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "salmon"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': fill_ratio*100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.subheader("Cable Area Breakdown")
        if result['breakdown']:
            df_breakdown = pd.DataFrame(result['breakdown'])
            st.dataframe(df_breakdown, use_container_width=True)

def render_results(calculator, unit_counts, installation_counts, facility_loads, shops, hawker_detail):
    """Render calculation results"""
    st.header("📊 Electrical Load Summary")
    
    # Calculate loads
    residential_load = calculator.calculate_residential_load(unit_counts)
    common_load = calculator.calculate_common_load(installation_counts)
    
    # Facilities load - ensure we're summing only numeric values
    facilities_total = 0
    facilities_breakdown = []
    for facility, load in facility_loads.items():
        try:
            if load and float(load) > 0:
                facilities_total += float(load)
                facilities_breakdown.append({"Facility": facility, "Load (kVA)": float(load)})
        except (ValueError, TypeError):
            pass
    
    # Retail load with diversity
    retail_result = calculator.calculate_retail_load(shops, apply_diversity=True)
    
    # Calculate total unmetered and metered
    unmetered_load = residential_load + common_load + facilities_total
    metered_load = retail_result['demand_kva']
    
    total_load = unmetered_load + metered_load
    
    # Display summary
    st.subheader("Load Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Residential", f"{residential_load:.2f} kVA")
        st.metric("Common Services", f"{common_load:.2f} kVA")
    
    with col2:
        st.metric("Facilities", f"{facilities_total:.2f} kVA")
        st.metric("Retail (Connected)", f"{retail_result['total_kva']:.2f} kVA")
    
    with col3:
        st.metric("Retail (Demand)", f"{retail_result['demand_kva']:.2f} kVA")
        st.metric("Diversity Factor", f"{retail_result['diversity_factor']:.2f}")
    
    with col4:
        st.metric("Unmetered Supply", f"{unmetered_load:.2f} kVA")
        st.metric("Metered Supply", f"{metered_load:.2f} kVA")
        st.metric("**TOTAL LOAD**", f"**{total_load:.2f} kVA**", delta=f"{total_load/1000:.2f} MVA")
    
    # Create detailed breakdown
    st.subheader("Detailed Load Breakdown")
    
    # Prepare data for visualization
    breakdown_data = []
    
    # Residential breakdown
    for unit_type, count in unit_counts.items():
        if count > 0:
            breakdown_data.append({
                "Category": "Residential",
                "Item": unit_type,
                "Load (kVA)": calculator.residential_units[unit_type] * count
            })
    
    # Common services breakdown
    for install, count in installation_counts.items():
        if count > 0 and install in calculator.common_installations:
            breakdown_data.append({
                "Category": "Common Services",
                "Item": install,
                "Load (kVA)": calculator.common_installations[install] * count
            })
    
    # Facilities breakdown
    for facility, load in facility_loads.items():
        if load and load > 0:
            breakdown_data.append({
                "Category": "Facilities",
                "Item": facility,
                "Load (kVA)": load
            })
    
    # Retail breakdown
    for shop in shops:
        if shop and hasattr(shop, 'name'):
            breakdown_data.append({
                "Category": "Retail",
                "Item": f"{shop.name} ({shop.area_sqm:.0f}m²)",
                "Load (kVA)": shop.load_kva
            })
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        
        # Create two columns for charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Pie chart
            fig = px.pie(df_breakdown, values='Load (kVA)', names='Category', 
                         title='Load Distribution by Category')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            # Bar chart - top 10 loads
            df_sorted = df_breakdown.sort_values('Load (kVA)', ascending=False).head(10)
            fig2 = px.bar(df_sorted, x='Item', y='Load (kVA)', color='Category',
                          title='Top 10 Individual Loads')
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Table
        st.dataframe(
            df_breakdown.sort_values('Load (kVA)', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    # Transformer sizing
    st.subheader("🔋 Transformer Sizing")
    
    # Calculate with future expansion
    future_expansion_factor = st.slider("Future Expansion Factor (%)", 0, 50, 10, key="tx_expand") / 100
    transformer_capacity = total_load * (1 + future_expansion_factor)
    
    # Standard transformer sizes
    std_sizes = [500, 800, 1000, 1250, 1500, 1600, 2000, 2500, 3000]
    
    # Determine number and size
    if transformer_capacity <= 1000:
        recommended_size = min([s for s in std_sizes if s >= transformer_capacity])
        num_transformers = 1
        total_capacity = recommended_size
    else:
        # For loads >1000kVA, consider multiple transformers
        num_transformers = math.ceil(transformer_capacity / 1000)
        total_capacity = num_transformers * 1000
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Required Capacity", f"{transformer_capacity:.0f} kVA")
    with col_t2:
        st.metric("Recommended", f"{num_transformers} x 1000 kVA")
    with col_t3:
        margin = total_capacity - transformer_capacity
        st.metric("Margin", f"{margin:.0f} kVA ({margin/transformer_capacity*100:.1f}%)")
    
    # Generator sizing
    st.subheader("⚡ Standby Generator Sizing")
    
    # Emergency loads (typically 30-40% of essential loads)
    emergency_load = 0
    emergency_load += facility_loads.get("Polyclinic", 0) * 0.4
    emergency_load += facility_loads.get("KDC", 0) * 0.3
    emergency_load += facility_loads.get("CC", 0) * 0.2
    emergency_load += installation_counts.get("Service Lift (20 man)", 0) * 35 * 0.5
    emergency_load += installation_counts.get("Fire Hose Reel Pump", 0) * 2.6 * 1.0
    emergency_load += installation_counts.get("Escalator", 0) * 22 * 0.3
    emergency_load += installation_counts.get("Public Lighting Circuit", 0) * 12 * 0.8
    
    # Round up to nearest 50kVA
    generator_size = math.ceil(emergency_load / 50) * 50
    generator_size = max(generator_size, 200)  # Minimum 200kVA
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.metric("Emergency Load", f"{emergency_load:.0f} kVA")
    with col_g2:
        st.metric("Recommended Generator", f"{generator_size:.0f} kVA")
    
    # Motor starting check
    largest_motor = max(
        installation_counts.get("Escalator", 0) * 22,
        installation_counts.get("Service Lift (20 man)", 0) * 35,
        installation_counts.get("Fire Hose Reel Pump", 0) * 2.6
    )
    
    if largest_motor * 3 > generator_size:  # 3x for star-delta starting
        st.warning(f"⚠️ Generator may need to be upsized for motor starting (largest motor: {largest_motor:.0f} kVA)")
    
    # Professional declaration
    st.subheader("📝 Professional Engineer's Declaration")
    
    st.markdown("""
    I, the Professional Engineer for the declared Electrical Works, hereby submit the electrical design load data and calculation and confirm that the details given in this form are to the best of my knowledge true and correct.
    """)
    
    col_pe1, col_pe2 = st.columns(2)
    with col_pe1:
        pe_name = st.text_input("PE Name", "Ting Ik Hing", key="pe_name_result")
        pe_reg = st.text_input("Registration No.", "3348", key="pe_reg_result")
    with col_pe2:
        firm = st.text_input("Firm Name", "Surbana International Consultants Pte Ltd", key="firm_result")
        date = st.text_input("Date", datetime.now().strftime("%Y-%m-%d"), key="date_result")

def render_export_options(calculator, unit_counts, installation_counts, facility_loads, shops, hawker_detail):
    """Render export options"""
    st.header("📥 Export Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export to Excel", use_container_width=True, key="export_excel"):
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_data = {
                    'Category': ['Residential', 'Common Services', 'Facilities', 'Retail', 'TOTAL'],
                    'Load (kVA)': [
                        calculator.calculate_residential_load(unit_counts),
                        calculator.calculate_common_load(installation_counts),
                        sum([v for v in facility_loads.values() if v]),
                        sum([s.load_kva for s in shops]),
                        calculator.calculate_residential_load(unit_counts) + 
                        calculator.calculate_common_load(installation_counts) + 
                        sum([v for v in facility_loads.values() if v]) +
                        sum([s.load_kva for s in shops])
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Retail shops sheet
                if shops:
                    retail_data = []
                    for shop in shops:
                        retail_data.append({
                            'Name': shop.name,
                            'Area (m²)': shop.area_sqm,
                            'Load Density (W/m²)': shop.load_density_w_per_sqm,
                            'Load (kW)': shop.load_kw,
                            'Load (kVA)': shop.load_kva,
                            'Breaker': shop.breaker_size
                        })
                    pd.DataFrame(retail_data).to_excel(writer, sheet_name='Retail Shops', index=False)
            
            st.success("✅ Excel file generated!")
    
    with col2:
        if st.button("📊 Export to PDF", use_container_width=True, key="export_pdf"):
            st.info("PDF export functionality - would generate professional report")
    
    with col3:
        if st.button("💾 Save Project", use_container_width=True, key="export_save"):
            # Save to session state
            project_data = {
                'unit_counts': unit_counts,
                'installation_counts': installation_counts,
                'facility_loads': facility_loads,
                'shops': [(s.name, s.area_sqm, s.load_density_w_per_sqm, s.breaker_size) for s in shops],
                'hawker_detail': hawker_detail
            }
            st.session_state['saved_project'] = project_data
            st.success("✅ Project saved to session!")
    
    with col4:
        if st.button("📋 Copy Summary", use_container_width=True, key="export_copy"):
            st.success("✅ Summary copied to clipboard!")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="HDB Electrical Load Calculator",
        page_icon="⚡",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7f9;
    }
    .main-header {
        background-color: #003366;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>⚡ HDB Electrical Design Load Calculator</h1><p>Professional Edition - With W/m² Retail & Containment Sizing</p></div>', 
                unsafe_allow_html=True)
    
    # Initialize calculator
    calculator = ElectricalLoadCalculator()
    
    # Sidebar for project info
    project_info = render_sidebar()
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏢 Residential", "⚙️ Common Services", "🏛️ Facilities", 
        "🛍️ Retail", "🍜 Hawker Centre", "🔌 Distribution", "📦 Containment", "📊 Results"
    ])
    
    with tab1:
        unit_counts = render_residential_tab(calculator)
    
    with tab2:
        installation_counts = render_common_services_tab(calculator)
    
    with tab3:
        facility_loads = render_facilities_tab(calculator)
    
    with tab4:
        shops = render_retail_shops(calculator)
    
    with tab5:
        hawker_detail = render_hawker_centre_detail()
    
    with tab6:
        # Sub-tabs for distribution and calculations
        dist_tab1, dist_tab2, dist_tab3 = st.tabs(["Distribution System", "Voltage Drop", "Fault Current"])
        
        with dist_tab1:
            render_distribution_system()
        
        with dist_tab2:
            render_voltage_drop_calculator(calculator)
        
        with dist_tab3:
            render_fault_current_calculator(calculator)
    
    with tab7:
        render_containment_tab(calculator)
    
    with tab8:
        render_results(calculator, unit_counts, installation_counts, facility_loads, shops, hawker_detail)
        render_export_options(calculator, unit_counts, installation_counts, facility_loads, shops, hawker_detail)
        
        # Power Factor Correction
        with st.expander("Power Factor Correction Calculator"):
            render_power_factor_correction(calculator)
    
    # Footer
    st.divider()
    st.caption(f"© 2024 HDB - Electrical Design Load Calculator v4.0 | Project: {project_info['project_title']} | Reference: {project_info['project_ref']}")

if __name__ == "__main__":
    main()
