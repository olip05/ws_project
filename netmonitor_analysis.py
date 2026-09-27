#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NetMonitor Analysis

Analyse von NetMonitor CLF + optional KML.

Ausgaben:
    01_route.png
    02_signal_map.png
    03_cell_changes.png
    04_signal_time.png
    05_signal_by_cell.png
    06_summary.png
    NetMonitor_Report.pdf

Aufruf:
    python netmonitor_analysis.py --clf 2.clf --kml 2.kml

Stadia API Key:
    PowerShell:
        $env:STADIA_API_KEY="DEIN_API_KEY"

    CMD:
        set STADIA_API_KEY=DEIN_API_KEY
"""

import argparse
import io
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import requests


# ============================================================
# KONFIGURATION
# ============================================================

FIGSIZE = (13.333, 7.5)
DPI = 150

SIGNAL_VMIN = -120
SIGNAL_VMAX = -50

# Stadia
STADIA_STYLE = "alidade_smooth"

# 1200 x 675 ergibt ein 16:9-Kartenbild.
# @2x liefert eine hochauflösende Karte.
STADIA_WIDTH = 1200
STADIA_HEIGHT = 675
STADIA_SCALE = "@2x"

# etwas Rand um die Messroute
MAP_PADDING = 0.00015

# Karten-Zoom
MIN_ZOOM = 14
MAX_ZOOM = 18

# KML
LABEL_KML_CELL_SITES = True

# Dateien
DEFAULT_OUTPUT_DIR = "netmonitor_output"


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def clean_text(value):
    """Text sicher bereinigen."""
    if value is None:
        return ""

    return str(value).strip()


def safe_float(value):
    """Float mit deutschem Dezimaltrennzeichen."""
    if value is None:
        return np.nan

    text = str(value).strip()

    if not text or text in ("-", "—", "–"):
        return np.nan

    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_signal(value):
    """
    NetMonitor-Signalwert.

    99 = kein Empfang.
    """
    value = clean_text(value)

    if value in ("", "-", "99"):
        return np.nan

    return safe_float(value)


def parse_distance(value):
    """
    Extrahiert z.B.

        ~1983 (m)

    zu:

        1983.0
    """
    value = clean_text(value)

    if not value or value == "-":
        return np.nan

    match = re.search(r"([\d.,]+)", value)

    if not match:
        return np.nan

    return safe_float(match.group(1))


def signal_color_map():
    return plt.get_cmap("turbo")


# ============================================================
# CLF
# ============================================================

def read_clf_file(path):
    """
    Liest das tatsächliche NetMonitor-CLF-Format.

    Beispiel:

    20260927  204108  1236227  29120  -  244  05
    +60,187608  +24,824344  -  -73  -  ~1983 (m)

    Logische Felder:

        0  Datum
        1  Uhrzeit
        2  Cell ID
        3  LAC/TAC
        4  -
        5  MCC
        6  MNC
        7  Latitude
        8  Longitude
        9  -
        10 Signal dBm
        11 -
        12 Zusatzinformationen
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CLF-Datei nicht gefunden: {path}"
        )

    records = []
    malformed = 0

    print(f"Lese CLF: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as handle:

        for line_no, line in enumerate(handle, start=1):

            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            # ----------------------------------------------------
            # WICHTIG:
            # CLF ist TAB-getrennt.
            # ----------------------------------------------------

            fields = line.split("\t")

            if len(fields) < 13:

                malformed += 1

                print(
                    f"Warnung: Zeile {line_no} hat nur "
                    f"{len(fields)} Felder."
                )

                continue

            try:

                # ------------------------------------------------
                # Zeit
                # ------------------------------------------------

                date_text = clean_text(fields[0])
                time_text = clean_text(fields[1])

                timestamp = pd.to_datetime(
                    f"{date_text} {time_text}",
                    format="%Y%m%d %H%M%S",
                    errors="coerce"
                )

                if pd.isna(timestamp):
                    malformed += 1
                    continue

                # ------------------------------------------------
                # Mobilfunk
                # ------------------------------------------------

                cell_id = clean_text(fields[2])
                lac_tac = clean_text(fields[3])

                mcc = clean_text(fields[5])
                mnc = clean_text(fields[6])

                # ------------------------------------------------
                # Position
                #
                # +60,187608 -> 60.187608
                # +24,824344 -> 24.824344
                # ------------------------------------------------

                latitude = safe_float(fields[7])
                longitude = safe_float(fields[8])

                has_position = (
                    not pd.isna(latitude)
                    and not pd.isna(longitude)
                    and -90 <= latitude <= 90
                    and -180 <= longitude <= 180
                )

                # ------------------------------------------------
                # Signal
                # ------------------------------------------------

                signal_dbm = parse_signal(fields[10])

                # ------------------------------------------------
                # Zusatzinformation
                # ------------------------------------------------

                extra = clean_text(fields[12])

                # ------------------------------------------------
                # Zusatzinformationen aus NetMonitor
                #
                # z.B.
                # ~490 (m), Uusimaa, Maarintie 8,
                # 02150 Espoo, Finnland
                # ------------------------------------------------

                distance_m = parse_distance(extra)

                # ------------------------------------------------
                # Verbindung
                # ------------------------------------------------

                connected = (
                    has_position
                    and cell_id not in ("", "0")
                    and not pd.isna(signal_dbm)
                )

                # ------------------------------------------------
                # Record
                # ------------------------------------------------

                records.append({
                    "timestamp": timestamp,

                    "cell_id": cell_id,
                    "lac_tac": lac_tac,

                    "mcc": mcc,
                    "mnc": mnc,

                    "latitude": latitude,
                    "longitude": longitude,

                    "signal_dbm": signal_dbm,

                    "connected": connected,
                    "has_position": has_position,

                    "extra": extra,
                    "distance_m": distance_m,
                })

            except Exception as exc:

                malformed += 1

                print(
                    f"Warnung: Fehler in CLF-Zeile "
                    f"{line_no}: {exc}"
                )

    if not records:
        raise ValueError(
            f"Keine gültigen Datensätze in {path} gefunden."
        )

    df = pd.DataFrame(records)

    df.sort_values(
        "timestamp",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # ------------------------------------------------------------
    # Statusänderungen
    # ------------------------------------------------------------

    df["prev_cell_id"] = df["cell_id"].shift(1)

    df["cell_change"] = (
        df["connected"]
        & df["prev_cell_id"].notna()
        & (df["cell_id"] != df["prev_cell_id"])
        & (df["prev_cell_id"] != "0")
    )

    df["connection_lost"] = (
        (~df["connected"])
        & df["connected"].shift(1).fillna(False)
    )

    df["connection_restored"] = (
        df["connected"]
        & (~df["connected"].shift(1).fillna(False))
    )

    print(
        f"  Datensätze: {len(df)}"
    )

    print(
        f"  Mit Position: "
        f"{int(df['has_position'].sum())}"
    )

    print(
        f"  Mit Verbindung: "
        f"{int(df['connected'].sum())}"
    )

    print(
        f"  Zellwechsel: "
        f"{int(df['cell_change'].sum())}"
    )

    print(
        f"  Verbindungsverluste: "
        f"{int(df['connection_lost'].sum())}"
    )

    if malformed:
        print(
            f"  Übersprungene Zeilen: {malformed}"
        )

    return df


# ============================================================
# KML
# ============================================================

KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}


def kml_coordinates(text):
    """
    Liest KML-Koordinaten.

    KML:

        longitude,latitude[,altitude]
    """

    result = []

    if not text:
        return result

    for item in text.strip().split():

        parts = item.split(",")

        if len(parts) < 2:
            continue

        try:

            lon = float(parts[0])
            lat = float(parts[1])

            result.append(
                (lat, lon)
            )

        except ValueError:
            continue

    return result


def parse_kml_change_type(description):
    """Liest CHANGE_TYPE aus einer KML-Beschreibung."""

    if not description:
        return ""

    match = re.search(
        r"CHANGE_TYPE:\s*([^\s<]+)",
        description,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


def parse_kml_timestamp(description):
    """Liest TIMESTAMP aus einer KML-Beschreibung."""

    if not description:
        return pd.NaT

    match = re.search(
        r"TIMESTAMP:\s*(.+)",
        description,
        flags=re.IGNORECASE
    )

    if not match:
        return pd.NaT

    text = match.group(1).strip()

    return pd.to_datetime(
        text,
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce"
    )


def parse_kml_cell_metadata(description):
    """
    Liest z.B.:

        LAC: 29120
        CID: 4618:4
        RNC: 0
        PSC: 333
    """

    result = {
        "lac": "",
        "cid": "",
        "rnc": "",
        "psc": "",
    }

    if not description:
        return result

    patterns = {
        "lac": r"LAC:\s*([^\s]+)",
        "cid": r"CID:\s*([^\s]+)",
        "rnc": r"RNC:\s*([^\s]+)",
        "psc": r"PSC:\s*([^\s]+)",
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            description,
            flags=re.IGNORECASE
        )

        if match:
            result[key] = match.group(1)

    return result


def read_kml_file(path):
    """
    Liest:
        - GPS Route
        - Cell Sites
        - CELL Changes
        - DBM Changes
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"KML-Datei nicht gefunden: {path}"
        )

    print(f"Lese KML: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    route = []
    cell_sites = []
    changes = []

    # ------------------------------------------------------------
    # Alle Placemarks
    # ------------------------------------------------------------

    for placemark in root.findall(
        ".//kml:Placemark",
        KML_NS
    ):

        name_element = placemark.find(
            "kml:name",
            KML_NS
        )

        name = (
            name_element.text.strip()
            if name_element is not None
            and name_element.text
            else ""
        )

        description_element = placemark.find(
            "kml:description",
            KML_NS
        )

        description = (
            description_element.text
            if description_element is not None
            and description_element.text
            else ""
        )

        # --------------------------------------------------------
        # LineString
        # --------------------------------------------------------

        line_coordinates = placemark.find(
            ".//kml:LineString/kml:coordinates",
            KML_NS
        )

        if line_coordinates is not None:

            coords = kml_coordinates(
                line_coordinates.text
            )

            if coords:
                route.extend(coords)

        # --------------------------------------------------------
        # Point
        # --------------------------------------------------------

        point_coordinates = placemark.find(
            ".//kml:Point/kml:coordinates",
            KML_NS
        )

        if point_coordinates is None:
            continue

        coords = kml_coordinates(
            point_coordinates.text
        )

        if not coords:
            continue

        latitude, longitude = coords[0]

        change_type = parse_kml_change_type(
            description
        )

        timestamp = parse_kml_timestamp(
            description
        )

        metadata = parse_kml_cell_metadata(
            description
        )

        # --------------------------------------------------------
        # Zellstandort
        # --------------------------------------------------------

        if (
            "LTE" in name.upper()
            or "GSM" in name.upper()
            or "UMTS" in name.upper()
            or "NR" in name.upper()
        ):

            cell_sites.append({
                "name": name,
                "description": description,
                "latitude": latitude,
                "longitude": longitude,
                **metadata,
            })

        # --------------------------------------------------------
        # Change
        # --------------------------------------------------------

        if change_type:

            signal = safe_float(name)

            changes.append({
                "timestamp": timestamp,
                "change_type": change_type,
                "signal_dbm": signal,
                "latitude": latitude,
                "longitude": longitude,
                "name": name,
                "description": description,
                **metadata,
            })

    print(
        f"  KML-Routenpunkte: {len(route)}"
    )

    print(
        f"  KML-Zellstandorte: {len(cell_sites)}"
    )

    print(
        f"  KML-Änderungen: {len(changes)}"
    )

    return {
        "route": route,
        "cell_sites": cell_sites,
        "changes": changes,
    }


# ============================================================
# KARTENBEREICH
# ============================================================

def calculate_map_bounds(clf, kml=None, padding_ratio=0.03):
    """
    Berechnet den Kartenausschnitt ausschließlich aus den
    tatsächlichen Mess-/GPS-Punkten.

    Basisstationen werden bewusst NICHT berücksichtigt.
    """

    # Nur echte GPS-Messpunkte aus der CLF
    valid = clf[clf["has_position"]].copy()

    if valid.empty:
        raise ValueError("Keine gültigen GPS-Messpunkte in der CLF-Datei gefunden.")

    lats = valid["latitude"].tolist()
    lons = valid["longitude"].tolist()

    # Falls eine KML vorhanden ist:
    # nur deren tatsächliche GPS-Route zusätzlich berücksichtigen,
    # NICHT die Cell Sites.
    if kml:
        for lat, lon in kml.get("route", []):
            lats.append(lat)
            lons.append(lon)

    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # 3 % Rand um die tatsächlichen Messdaten
    lat_padding = max(lat_range * padding_ratio, 0.00005)
    lon_padding = max(lon_range * padding_ratio, 0.00005)

    return (
        min_lat - lat_padding,
        max_lat + lat_padding,
        min_lon - lon_padding,
        max_lon + lon_padding,
    )

def calculate_zoom(
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    width=STADIA_WIDTH,
    height=STADIA_HEIGHT,
):
    """
    Berechnet die maximale Stadia-Zoomstufe, bei der
    die komplette Bounding Box in den 16:9-Viewport passt.
    """

    lon_range = max_lon - min_lon

    lat1 = math.radians(
        max(-85.05112878, min(85.05112878, min_lat))
    )

    lat2 = math.radians(
        max(-85.05112878, min(85.05112878, max_lat))
    )

    # Horizontale Mercator-Spanne
    if lon_range <= 0:
        zoom_lon = MAX_ZOOM
    else:
        zoom_lon = math.log2(
            360.0 * width /
            (256.0 * lon_range)
        )

    # Vertikale Mercator-Spanne
    y1 = math.log(
        math.tan(math.pi / 4 + lat1 / 2)
    )

    y2 = math.log(
        math.tan(math.pi / 4 + lat2 / 2)
    )

    y_range = abs(y2 - y1)

    if y_range <= 0:
        zoom_lat = MAX_ZOOM
    else:
        zoom_lat = math.log2(
            2 * math.pi * height /
            (256.0 * y_range)
        )

    zoom = min(
        zoom_lon,
        zoom_lat
    )

    return max(
        MIN_ZOOM,
        min(MAX_ZOOM, int(math.floor(zoom)))
    )

def calculate_viewport_bounds(
    center_lat,
    center_lon,
    zoom,
    width=STADIA_WIDTH,
    height=STADIA_HEIGHT,
):
    """
    Berechnet die tatsächlichen geografischen Grenzen des
    Stadia-Static-Map-Bildes.

    Wichtig:
    Diese Grenzen werden später für imshow(extent=...)
    verwendet. Dadurch stimmen Stadia-Karte und Messpunkte
    geometrisch überein.
    """

    world_size = 256.0 * (2 ** zoom)

    # --------------------------------------------------------
    # Mittelpunkt in Web Mercator Pixelkoordinaten
    # --------------------------------------------------------

    center_x = (
        (center_lon + 180.0) / 360.0
    ) * world_size

    lat_rad = math.radians(center_lat)

    center_y = (
        (
            1.0
            - math.asinh(math.tan(lat_rad)) / math.pi
        )
        / 2.0
    ) * world_size

    # --------------------------------------------------------
    # Viewport-Größe
    #
    # @2x ändert die Pixelauflösung,
    # aber NICHT den geografischen Ausschnitt.
    # --------------------------------------------------------

    half_width = width / 2.0
    half_height = height / 2.0

    left_x = center_x - half_width
    right_x = center_x + half_width

    top_y = center_y - half_height
    bottom_y = center_y + half_height

    # --------------------------------------------------------
    # Web Mercator -> Longitude
    # --------------------------------------------------------

    min_lon = (
        left_x / world_size * 360.0
    ) - 180.0

    max_lon = (
        right_x / world_size * 360.0
    ) - 180.0

    # --------------------------------------------------------
    # Web Mercator -> Latitude
    # --------------------------------------------------------

    def pixel_y_to_lat(y):

        n = math.pi - (
            2.0 * math.pi * y / world_size
        )

        return math.degrees(
            math.atan(
                math.sinh(n)
            )
        )

    max_lat = pixel_y_to_lat(top_y)
    min_lat = pixel_y_to_lat(bottom_y)

    return (
        min_lat,
        max_lat,
        min_lon,
        max_lon,
    )

# ============================================================
# STADIA MAPS
# ============================================================

def get_stadia_api_key():
    """
    Liest STADIA_API_KEY aus der Umgebung.
    """

    key = os.environ.get(
        "STADIA_API_KEY"
    )

    if not key:

        raise RuntimeError(
            "STADIA_API_KEY ist nicht gesetzt.\n\n"
            "PowerShell:\n"
            '$env:STADIA_API_KEY="DEIN_API_KEY"\n\n'
            "Danach das Skript erneut starten."
        )

    return key.strip()


def download_stadia_map(
    clf,
    kml,
):
    """
    Lädt eine einzige Stadia Static Map.

    Diese Karte wird anschließend für alle Kartenplots
    wiederverwendet.
    """

    api_key = get_stadia_api_key()

    min_lat, max_lat, min_lon, max_lon = (
        calculate_map_bounds(
            clf,
            kml
        )
    )

    center_lat = (
        min_lat + max_lat
    ) / 2

    center_lon = (
        min_lon + max_lon
    ) / 2

    zoom = calculate_zoom(
        min_lat,
        max_lat,
        min_lon,
        max_lon,
    )

    base_url = (
        f"https://tiles.stadiamaps.com/"
        f"static/{STADIA_STYLE}.png"
    )

    params = {
        "center": (
            f"{center_lat:.6f},"
            f"{center_lon:.6f}"
        ),
        "zoom": zoom,
        "size": (
            f"{STADIA_WIDTH}x"
            f"{STADIA_HEIGHT}"
            f"{STADIA_SCALE}"
        ),
        "dpi": 150,
    }

    headers = {
        "Authorization":
            f"Stadia-Auth {api_key}"
    }

    print(
        "Lade Stadia-Karte..."
    )

    print(
        f"  Zentrum: "
        f"{center_lat:.6f}, "
        f"{center_lon:.6f}"
    )

    print(
        f"  Zoom: {zoom}"
    )

    try:

        response = requests.get(
            base_url,
            params=params,
            headers=headers,
            timeout=60,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Fehler beim Abrufen der Stadia-Karte: {exc}"
        ) from exc

    if response.status_code != 200:

        body = response.text[:500]

        raise RuntimeError(
            "Stadia Static Maps API Fehler:\n"
            f"HTTP {response.status_code}\n"
            f"{body}"
        )

    try:

        image = plt.imread(
            io.BytesIO(
                response.content
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Stadia-Kartenbild konnte nicht gelesen werden: {exc}"
        ) from exc

    viewport_bounds = calculate_viewport_bounds(
        center_lat,
        center_lon,
        zoom,
    )

    return {
        "image": image,

        # NICHT mehr die Daten-Bounding-Box,
        # sondern der tatsächliche Stadia-Viewport
        "bounds": viewport_bounds,

        "data_bounds": (
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        ),

        "center": (
            center_lat,
            center_lon,
        ),

        "zoom": zoom,
    }


# ============================================================
# KARTEN-GRUNDLAGE
# ============================================================

def setup_map_axis(
    ax,
    basemap,
):
    """Zeigt die Stadia-Karte an."""

    min_lat, max_lat, min_lon, max_lon = (
        basemap["bounds"]
    )

    ax.imshow(
        basemap["image"],
        extent=[
            min_lon,
            max_lon,
            min_lat,
            max_lat,
        ],
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )

    ax.set_xlim(
        min_lon,
        max_lon
    )

    ax.set_ylim(
        min_lat,
        max_lat
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.5,
    )


# ============================================================
# ROUTE
# ============================================================

def plot_route(
    clf,
    kml,
    basemap,
    output_path,
):
    """GPS-Route."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        dpi=DPI
    )

    setup_map_axis(
        ax,
        basemap
    )

    valid = clf[
        clf["has_position"]
    ]

    ax.plot(
        valid["longitude"],
        valid["latitude"],
        linewidth=2.0,
        color="black",
        zorder=5,
        label="CLF GPS-Route",
    )

    # KML Route
    route = kml.get(
        "route",
        []
    )

    if route:

        route_lats = [
            x[0] for x in route
        ]

        route_lons = [
            x[1] for x in route
        ]

        ax.plot(
            route_lons,
            route_lats,
            linewidth=1.2,
            color="yellow",
            alpha=0.85,
            zorder=4,
            label="KML GPS-Route",
        )

    # Start / Ende
    if not valid.empty:

        first = valid.iloc[0]
        last = valid.iloc[-1]

        ax.scatter(
            first["longitude"],
            first["latitude"],
            s=100,
            marker="o",
            color="green",
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
            label="Start",
        )

        ax.scatter(
            last["longitude"],
            last["latitude"],
            s=120,
            marker="X",
            color="red",
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
            label="Ende",
        )

    ax.set_title(
        "NetMonitor – GPS Route"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SIGNALKARTE
# ============================================================

def plot_signal_map(
    clf,
    kml,
    basemap,
    output_path,
):
    """Signalstärke als farbige Punkte."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        dpi=DPI
    )

    setup_map_axis(
        ax,
        basemap
    )

    data = clf[
        clf["signal_dbm"].notna()
        & clf["has_position"]
    ].copy()

    if not data.empty:

        norm = Normalize(
            vmin=SIGNAL_VMIN,
            vmax=SIGNAL_VMAX
        )

        scatter = ax.scatter(
            data["longitude"],
            data["latitude"],
            c=data["signal_dbm"],
            cmap=signal_color_map(),
            norm=norm,
            s=22,
            alpha=0.9,
            edgecolors="none",
            zorder=5,
        )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            pad=0.02
        )

        cbar.set_label(
            "Signalstärke [dBm]"
        )

    # --------------------------------------------------------
    # KML Cell Sites
    # --------------------------------------------------------

    sites = kml.get(
        "cell_sites",
        []
    )

    if sites:

        ax.scatter(
            [x["longitude"] for x in sites],
            [x["latitude"] for x in sites],
            marker="D",
            s=100,
            facecolors="none",
            edgecolors="white",
            linewidths=2,
            zorder=10,
        )

        if LABEL_KML_CELL_SITES:

            for site in sites:

                ax.annotate(
                    site["name"],
                    (
                        site["longitude"],
                        site["latitude"]
                    ),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="black",
                        alpha=0.6,
                    ),
                    zorder=11,
                )

    ax.set_title(
        "NetMonitor – Signalstärke"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# CELL CHANGES
# ============================================================

def plot_cell_changes(
    clf,
    kml,
    basemap,
    output_path,
):
    """Zellwechsel und Verbindungsverluste."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        dpi=DPI
    )

    setup_map_axis(
        ax,
        basemap
    )

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    data = clf[
        clf["has_position"]
    ]

    ax.plot(
        data["longitude"],
        data["latitude"],
        color="black",
        linewidth=1.5,
        alpha=0.7,
        zorder=3,
    )

    # --------------------------------------------------------
    # Cell Changes
    # --------------------------------------------------------

    changes = clf[
        clf["cell_change"]
        & clf["has_position"]
    ]

    if not changes.empty:

        ax.scatter(
            changes["longitude"],
            changes["latitude"],
            marker="*",
            s=180,
            color="orange",
            edgecolor="black",
            linewidth=1,
            zorder=10,
            label="Cell-Wechsel",
        )

        for _, row in changes.iterrows():

            ax.annotate(
                str(row["cell_id"]),
                (
                    row["longitude"],
                    row["latitude"]
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                ),
                zorder=11,
            )

    # --------------------------------------------------------
    # Connection Lost
    # --------------------------------------------------------

    lost = clf[
        clf["connection_lost"]
        & clf["has_position"]
    ]

    if not lost.empty:

        ax.scatter(
            lost["longitude"],
            lost["latitude"],
            marker="x",
            s=130,
            color="red",
            linewidth=2.5,
            zorder=12,
            label="Verbindung verloren",
        )

    # --------------------------------------------------------
    # Connection Restored
    # --------------------------------------------------------

    restored = clf[
        clf["connection_restored"]
        & clf["has_position"]
    ]

    if not restored.empty:

        ax.scatter(
            restored["longitude"],
            restored["latitude"],
            marker="o",
            s=80,
            facecolors="none",
            edgecolors="green",
            linewidth=2,
            zorder=11,
            label="Verbindung wiederhergestellt",
        )

    # --------------------------------------------------------
    # KML CELL changes
    # --------------------------------------------------------

    kml_changes = [
        x for x in kml.get(
            "changes",
            []
        )
        if x["change_type"].upper() == "CELL"
    ]

    if kml_changes:

        ax.scatter(
            [x["longitude"] for x in kml_changes],
            [x["latitude"] for x in kml_changes],
            marker="D",
            s=55,
            color="cyan",
            edgecolor="black",
            linewidth=0.7,
            zorder=13,
            label="KML CELL Change",
        )

    ax.set_title(
        "NetMonitor – Zellwechsel und Verbindungsverluste"
    )

    handles, labels = ax.get_legend_handles_labels()

    if handles:

        ax.legend(
            handles,
            labels,
            loc="best"
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SIGNAL TIME
# ============================================================

def plot_signal_time(
    clf,
    output_path,
):
    """Signalstärke über der Zeit."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        dpi=DPI
    )

    data = clf[
        clf["signal_dbm"].notna()
    ].copy()

    if data.empty:

        ax.text(
            0.5,
            0.5,
            "Keine Signalwerte vorhanden.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    else:

        ax.plot(
            data["timestamp"],
            data["signal_dbm"],
            linewidth=1.2,
            color="black",
        )

        ax.scatter(
            data["timestamp"],
            data["signal_dbm"],
            c=data["signal_dbm"],
            cmap=signal_color_map(),
            norm=Normalize(
                vmin=SIGNAL_VMIN,
                vmax=SIGNAL_VMAX
            ),
            s=10,
            zorder=3,
        )

        # Cell changes
        changes = clf[
            clf["cell_change"]
            & clf["signal_dbm"].notna()
        ]

        if not changes.empty:

            ax.scatter(
                changes["timestamp"],
                changes["signal_dbm"],
                marker="*",
                s=130,
                color="orange",
                edgecolor="black",
                zorder=10,
            )

        ax.set_ylabel(
            "Signal [dBm]"
        )

        ax.set_xlabel(
            "Zeit"
        )

        ax.grid(
            True,
            alpha=0.3
        )

    ax.set_title(
        "NetMonitor – Signalstärke über Zeit"
    )

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SIGNAL PRO CELL
# ============================================================

def plot_signal_by_cell(
    clf,
    output_path,
):
    """Signalverteilung pro Cell ID."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        dpi=DPI
    )

    data = clf[
        clf["connected"]
        & clf["signal_dbm"].notna()
    ].copy()

    if data.empty:

        ax.text(
            0.5,
            0.5,
            "Keine Signalwerte vorhanden.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    else:

        groups = []

        labels = []

        for cell_id, group in data.groupby(
            "cell_id"
        ):

            values = group[
                "signal_dbm"
            ].dropna().values

            if len(values) == 0:
                continue

            groups.append(values)
            labels.append(str(cell_id))

        if groups:

            bp = ax.boxplot(
                groups,
                patch_artist=False,
            )

            ax.set_xticks(
                range(1, len(labels) + 1)
            )

            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right"
            )

            ax.set_ylabel(
                "Signal [dBm]"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

        else:

            ax.text(
                0.5,
                0.5,
                "Keine Signalwerte vorhanden.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    ax.set_title(
        "NetMonitor – Signalverteilung pro Cell ID"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    clf,
    kml,
    output_path,
):
    """Erzeugt eine Zusammenfassung als PNG."""

    fig = plt.figure(
        figsize=FIGSIZE,
        dpi=DPI
    )

    ax = fig.add_subplot(
        111
    )

    ax.axis(
        "off"
    )

    total = len(clf)

    positions = int(
        clf["has_position"].sum()
    )

    connected = int(
        clf["connected"].sum()
    )

    no_connection = total - connected

    cell_changes = int(
        clf["cell_change"].sum()
    )

    connection_lost = int(
        clf["connection_lost"].sum()
    )

    restored = int(
        clf["connection_restored"].sum()
    )

    signal = clf[
        "signal_dbm"
    ].dropna()

    if len(signal):

        signal_min = signal.min()
        signal_max = signal.max()
        signal_mean = signal.mean()
        signal_median = signal.median()

    else:

        signal_min = np.nan
        signal_max = np.nan
        signal_mean = np.nan
        signal_median = np.nan

    cells = (
        clf.loc[
            clf["connected"],
            "cell_id"
        ]
        .value_counts()
    )

    start_time = clf[
        "timestamp"
    ].min()

    end_time = clf[
        "timestamp"
    ].max()

    duration = (
        end_time - start_time
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    lines = [
        "NetMonitor Analyse",
        "",
        f"Zeitraum:",
        f"  {start_time} – {end_time}",
        f"  Dauer: {duration}",
        "",
        f"Datensätze:           {total}",
        f"GPS-Positionen:       {positions}",
        f"Verbunden:             {connected}",
        f"Keine Verbindung:     {no_connection}",
        "",
        f"Zellwechsel:           {cell_changes}",
        f"Verbindungsverluste:   {connection_lost}",
        f"Wieder verbunden:      {restored}",
        "",
        "Signal:",
        f"  Minimum:             {signal_min:.1f} dBm"
        if not pd.isna(signal_min)
        else "  Minimum:             -",
        f"  Maximum:             {signal_max:.1f} dBm"
        if not pd.isna(signal_max)
        else "  Maximum:             -",
        f"  Mittelwert:          {signal_mean:.1f} dBm"
        if not pd.isna(signal_mean)
        else "  Mittelwert:          -",
        f"  Median:              {signal_median:.1f} dBm"
        if not pd.isna(signal_median)
        else "  Median:              -",
        "",
        f"KML GPS-Punkte:        {len(kml.get('route', []))}",
        f"KML Zellstandorte:     {len(kml.get('cell_sites', []))}",
        f"KML Änderungen:        {len(kml.get('changes', []))}",
        "",
        "Cell IDs:",
    ]

    for cell_id, count in cells.items():

        lines.append(
            f"  {cell_id}: {count} Messungen"
        )

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        family="monospace",
    )

    ax.set_title(
        "NetMonitor – Zusammenfassung",
        fontsize=18,
        pad=20,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# PDF
# ============================================================

def create_pdf(
    image_paths,
    output_path,
):
    """Erzeugt einen gemeinsamen PDF-Bericht."""

    print(
        f"Erzeuge PDF: {output_path}"
    )

    with PdfPages(
        output_path
    ) as pdf:

        for image_path in image_paths:

            if not Path(
                image_path
            ).exists():

                continue

            image = plt.imread(
                image_path
            )

            fig = plt.figure(
                figsize=FIGSIZE
            )

            ax = fig.add_axes(
                [0, 0, 1, 1]
            )

            ax.imshow(
                image
            )

            ax.axis(
                "off"
            )

            pdf.savefig(
                fig,
                dpi=DPI
            )

            plt.close(fig)


# ============================================================
# ARGUMENTE
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "NetMonitor CLF/KML Analyse "
            "mit Stadia Maps"
        )
    )

    parser.add_argument(
        "--clf",
        required=True,
        help="NetMonitor CLF-Datei"
    )

    parser.add_argument(
        "--kml",
        required=False,
        help="Optionale NetMonitor KML-Datei"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Ausgabeordner "
            f"(Standard: {DEFAULT_OUTPUT_DIR})"
        )
    )

    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Keine Stadia-Karte laden"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    clf_path = Path(
        args.clf
    )

    output_dir = Path(
        args.output + "/" + clf_path.name.removesuffix(".clf")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CLF
    # --------------------------------------------------------

    try:

        clf = read_clf_file(
            clf_path
        )

    except Exception as exc:

        print(
            f"Fehler beim Lesen der CLF-Datei: {exc}"
        )

        return 1

    # --------------------------------------------------------
    # KML
    # --------------------------------------------------------

    kml = {
        "route": [],
        "cell_sites": [],
        "changes": [],
    }

    if args.kml:

        try:

            kml = read_kml_file(
                args.kml
            )

        except Exception as exc:

            print(
                f"Warnung: KML konnte nicht gelesen werden: {exc}"
            )

    # --------------------------------------------------------
    # Stadia Karte
    # --------------------------------------------------------

    basemap = None

    if not args.no_map:

        try:

            basemap = download_stadia_map(
                clf,
                kml
            )

        except Exception as exc:

            print()
            print(
                "FEHLER beim Laden der Stadia-Karte:"
            )
            print(
                exc
            )
            print()
            print(
                "Die Analyse wird ohne Kartenhintergrund fortgesetzt."
            )
            print()

    # --------------------------------------------------------
    # Kartenplots
    # --------------------------------------------------------

    image_paths = []

    if basemap is not None:

        path = output_dir / "01_route.png"

        plot_route(
            clf,
            kml,
            basemap,
            path
        )

        image_paths.append(path)

        path = output_dir / "02_signal_map.png"

        plot_signal_map(
            clf,
            kml,
            basemap,
            path
        )

        image_paths.append(path)

        path = output_dir / "03_cell_changes.png"

        plot_cell_changes(
            clf,
            kml,
            basemap,
            path
        )

        image_paths.append(path)

    else:

        print(
            "Kartenplots werden übersprungen."
        )

    # --------------------------------------------------------
    # Signal Zeit
    # --------------------------------------------------------

    path = output_dir / "04_signal_time.png"

    plot_signal_time(
        clf,
        path
    )

    image_paths.append(path)

    # --------------------------------------------------------
    # Signal pro Cell
    # --------------------------------------------------------

    path = output_dir / "05_signal_by_cell.png"

    plot_signal_by_cell(
        clf,
        path
    )

    image_paths.append(path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    path = output_dir / "06_summary.png"

    make_summary(
        clf,
        kml,
        path
    )

    image_paths.append(path)

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    pdf_path = (
        output_dir /
        "NetMonitor_Report.pdf"
    )

    create_pdf(
        image_paths,
        pdf_path
    )

    # --------------------------------------------------------
    # Fertig
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ANALYSE FERTIG")
    print("=" * 60)
    print()
    print(
        f"Ausgabeordner: {output_dir.resolve()}"
    )
    print()

    for path in image_paths:

        print(
            f"  {path.name}"
        )

    print(
        f"  {pdf_path.name}"
    )

    print()

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )