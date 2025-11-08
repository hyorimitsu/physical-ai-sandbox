terraform {
  required_version = ">= 1.9.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

# -----------------------------
# Variables
# -----------------------------

variable "project_id" {
  type        = string
  description = "ID of the Google Cloud project to deploy into."
}

variable "region" {
  type        = string
  description = "Region for the resources."
  default     = "asia-northeast1"
}

variable "zone" {
  type        = string
  description = "Zone for the compute instance."
  default     = "asia-northeast1-a"
}

variable "instance_name" {
  type        = string
  description = "Name of the compute instance."
  default     = "physical-ai"
}

variable "machine_type" {
  type        = string
  description = "Machine type for the instance."
  default     = "g2-standard-4"
}

variable "boot_disk_size_gb" {
  type        = number
  description = "Size of the boot disk in GB."
  default     = 100
}

variable "boot_disk_type" {
  type        = string
  description = "Type of the boot disk."
  default     = "pd-balanced"
}

variable "boot_image" {
  type        = string
  description = "Boot disk image (image or image self link)."
  default     = "projects/ubuntu-os-cloud/global/images/ubuntu-2404-noble-amd64-v20251021"
}

variable "gpu_type" {
  type        = string
  description = "GPU accelerator type name (without project/zone prefix)."
  default     = "nvidia-l4"
}

variable "gpu_count" {
  type        = number
  description = "Number of GPUs to attach."
  default     = 1
}

# -----------------------------
# Provider
# -----------------------------

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# -----------------------------
# GPU VM instance
# -----------------------------

resource "google_compute_instance" "physical_ai" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    auto_delete = true

    initialize_params {
      image = var.boot_image
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
    }
  }

  guest_accelerator {
    count = var.gpu_count
    type  = "projects/${var.project_id}/zones/${var.zone}/acceleratorTypes/${var.gpu_type}"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  network_interface {
    network = "default"

    access_config {
      network_tier = "PREMIUM"
    }
  }

  metadata = {
    enable-osconfig = "TRUE"
  }

  labels = {
    env  = "sandbox"
    role = var.instance_name
  }

  service_account {
    email = null

    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }
}
