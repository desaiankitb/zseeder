"""
Pydantic models for structured prescription data extraction.
Supports eRx, fax, and transfer prescription formats.
"""

from typing import Optional, List, Literal
from datetime import date
from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Patient demographic and contact information."""
    name: Optional[str] = Field(None, description="Patient full name")
    date_of_birth: Optional[str] = Field(None, description="Patient date of birth (MM/DD/YYYY)")
    address: Optional[str] = Field(None, description="Patient street address")
    phone_numbers: Optional[List[str]] = Field(default_factory=list, description="Patient phone numbers")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Patient allergies")
    
    # Insurance information
    insurance_bin: Optional[str] = Field(None, description="Insurance BIN number")
    insurance_pcn: Optional[str] = Field(None, description="Insurance PCN")
    insurance_group: Optional[str] = Field(None, description="Insurance Rx Group")
    insurance_id: Optional[str] = Field(None, description="Insurance ID")
    insurance_person_code: Optional[str] = Field(None, description="Insurance person code")
    
    # Health demographics (optional)
    height: Optional[str] = Field(None, description="Patient height")
    weight: Optional[str] = Field(None, description="Patient weight")
    bmi: Optional[str] = Field(None, description="Patient BMI")


class PrescriberInfo(BaseModel):
    """Prescriber/Healthcare provider information."""
    name: Optional[str] = Field(None, description="Prescriber full name")
    address: Optional[str] = Field(None, description="Prescriber address")
    phone_number: Optional[str] = Field(None, description="Prescriber phone number")
    fax_number: Optional[str] = Field(None, description="Prescriber fax number")
    dea_number: Optional[str] = Field(None, description="DEA number")
    npi_number: Optional[str] = Field(None, description="NPI number")
    
    # Optional additional info
    practice_name: Optional[str] = Field(None, description="Practice or clinic name")
    specialty: Optional[str] = Field(None, description="Medical specialty")
    prescribing_agent: Optional[str] = Field(None, description="Agent prescribing on behalf")
    supervising_prescriber: Optional[str] = Field(None, description="Supervising prescriber if applicable")


class PrescriptionDetails(BaseModel):
    """Core prescription medication details."""
    drug_name: Optional[str] = Field(None, description="Medication name (e.g., Zepbound)")
    strength: Optional[str] = Field(None, description="Medication strength (e.g., 2.5mg, 5mg)")
    dosage_form: Optional[str] = Field(None, description="Form: vial, auto-injector, pen, etc.")
    directions: Optional[str] = Field(None, description="Directions for use/sig")
    quantity: Optional[str] = Field(None, description="Quantity prescribed")
    refills: Optional[str] = Field(None, description="Number of refills authorized")
    date_of_issue: Optional[str] = Field(None, description="Date prescription was written")
    daw_status: Optional[str] = Field(None, description="Dispense As Written status")
    
    # Clinical information
    diagnosis_codes: Optional[List[str]] = Field(default_factory=list, description="ICD diagnosis codes")
    prescriber_notes: Optional[str] = Field(None, description="Prescriber comments or notes")


class PharmacyInfo(BaseModel):
    """Pharmacy information (for transfers)."""
    pharmacy_name: Optional[str] = Field(None, description="Pharmacy name")
    store_number: Optional[str] = Field(None, description="Store or location number")
    address: Optional[str] = Field(None, description="Pharmacy address")
    phone_number: Optional[str] = Field(None, description="Pharmacy phone number")
    fax_number: Optional[str] = Field(None, description="Pharmacy fax number")
    dea_number: Optional[str] = Field(None, description="Pharmacy DEA number")


class TransferInfo(BaseModel):
    """Transfer-specific prescription information."""
    rx_number: Optional[str] = Field(None, description="Original Rx number from transferring pharmacy")
    original_refills: Optional[str] = Field(None, description="Original number of refills prescribed")
    refills_remaining: Optional[str] = Field(None, description="Number of refills remaining")
    fill_history: Optional[List[str]] = Field(default_factory=list, description="Previous fill dates")
    
    # Transfer details
    transferring_pharmacy: Optional[PharmacyInfo] = Field(None, description="Originating pharmacy info")
    receiving_pharmacy: Optional[PharmacyInfo] = Field(None, description="Receiving pharmacy info")
    transferring_pharmacist: Optional[str] = Field(None, description="Transferring pharmacist name")
    receiving_pharmacist: Optional[str] = Field(None, description="Receiving pharmacist name")
    transfer_date: Optional[str] = Field(None, description="Date of transfer")


class Prescription(BaseModel):
    """Complete prescription record."""
    prescription_type: Optional[Literal["eRx", "fax", "transfer"]] = Field(
        None, description="Type of prescription document"
    )
    
    patient: PatientInfo = Field(default_factory=PatientInfo, description="Patient information")
    prescriber: PrescriberInfo = Field(default_factory=PrescriberInfo, description="Prescriber information")
    prescription: List[PrescriptionDetails] = Field(default_factory=list, description="List of prescription details (multiple medications)")
    transfer: Optional[TransferInfo] = Field(None, description="Transfer information (if applicable)")
    
    # Validation metadata
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")
    is_valid: bool = Field(False, description="Overall validity status")
    validation_notes: Optional[str] = Field(None, description="Additional validation notes")


class PrescriptionBatch(BaseModel):
    """Container for multiple prescriptions from a single document."""
    prescriptions: List[Prescription] = Field(default_factory=list, description="List of prescriptions")
    document_name: str = Field(..., description="Source document filename")
    processed_date: str = Field(..., description="Date processed")
    total_count: int = Field(0, description="Total number of prescriptions")
