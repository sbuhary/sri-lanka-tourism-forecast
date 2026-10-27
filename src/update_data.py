"""Download SLTDA tables, check the counts and prepare a data update."""

from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit, quote
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import math
import re
import time

import openpyxl
import pandas as pd

folder = Path(__file__).resolve().parents[1]
source_page = 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries'
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
columns = ['Year_Month', 'Country', 'lag_1', 'lag_2', 'lag_3',
           'lag_12', 'month_number', 'Arrivals']


def normalise(value):
    return re.sub(r'\s+', ' ', str(value)).strip().lower().replace('&', 'and')


country_names = json.loads((folder / 'data/country_names.json').read_text())


def canonical_country(value):
    name = normalise(value)
    short = name.split(',')[0].strip()
    name = country_names.get(name, country_names.get(short, short)).title()
    for word in ['And', 'Of', 'The']:
        name = name.replace(f' {word} ', f' {word.lower()} ')
    return name.replace("Cote D'Ivoire", "Cote d'Ivoire")


def arrival_count(value):
    if value is None or str(value).strip() in ['', '-', '..', '...']:
        return None
    try:
        count = float(str(value).replace(',', '').strip())
    except ValueError as error:
        raise ValueError(f'Unexpected arrival count: {value!r}') from error
    if not math.isfinite(count) or count < 0 or not count.is_integer():
        raise ValueError(f'Invalid arrival count: {value!r}')
    return int(count)


class ExcelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.href = dict(attrs).get('href')
            self.label = []

    def handle_data(self, data):
        if self.href:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == 'a' and self.href:
            self.links.append((self.href, ' '.join(self.label)))
            self.href = None


def discover_files(html):
    parser = ExcelLinks()
    parser.feed(html)
    found = {}
    for href, label in parser.links:
        url = quote(urljoin(source_page, href), safe=':/%?=&')
        if not urlsplit(url).path.lower().endswith('.xlsx'):
            continue
        if 'all countries' not in normalise(label):
            continue
        year_match = re.search(r'\b(20\d{2})\b', label)
        if not year_match:
            raise ValueError(f'Cannot identify workbook year: {label}')
        year = int(year_match[1])
        if year < 2018:
            continue
        if year in found and found[year] != url:
            raise ValueError(f'Multiple Excel tables for {year}; review source links')
        found[year] = url
    if not found or set(range(2018, max(found) + 1)) != set(found):
        raise ValueError('SLTDA links are missing one or more years from 2018')
    return found


def download(url):
    if urlsplit(url).scheme != 'https' or urlsplit(url).hostname not in {
            'sltda.gov.lk', 'www.sltda.gov.lk'}:
        raise ValueError(f'Unexpected source URL: {url}')
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'TourismForecast/1.0 (SLTDA data update)'})
            with urlopen(request, timeout=60) as response:
                if urlsplit(response.url).hostname not in {'sltda.gov.lk', 'www.sltda.gov.lk'}:
                    raise ValueError('Download redirected outside SLTDA')
                content = response.read(20_000_001)
            if len(content) > 20_000_000:
                raise ValueError('SLTDA file exceeds the 20 MB download limit')
            return content
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_workbook(content, year):
    book = openpyxl.load_workbook(BytesIO(content), data_only=True, read_only=True)
    try:
        for sheet in book:
            rows = list(sheet.values)
            for row_number, row in enumerate(rows):
                labels = [normalise(value) for value in row]
                if 'country' not in labels:
                    continue
                country_column = labels.index('country')
                month_columns = {months.index(value[:3]) + 1: index
                                 for index, value in enumerate(labels) if value[:3] in months}
                if not month_columns:
                    continue
                if sorted(month_columns) != list(range(1, max(month_columns) + 1)):
                    raise ValueError(f'{year}: missing month headings')
                total_column = next((i for i, value in enumerate(labels)
                                     if value.startswith('total')), None)
                if total_column is None:
                    raise ValueError(f'{year}: no total column')
                entries, totals = [], None
                for line, values in enumerate(rows[row_number + 1:], start=row_number + 2):
                    if any(normalise(value) in ['total', 'grand total']
                           for value in values[:country_column + 1]):
                        if totals is not None:
                            raise ValueError(f'{year}: multiple total rows')
                        totals = values
                        continue
                    rank = values[country_column - 1]
                    if not isinstance(rank, (int, float)) or values[country_column] is None:
                        continue
                    source = str(values[country_column]).strip()
                    counts = {month: arrival_count(values[index])
                              for month, index in month_columns.items()}
                    annual = arrival_count(values[total_column])
                    entries.append({'source': source, 'country': canonical_country(source),
                                    'counts': counts, 'annual': annual, 'row': line})
                if totals is None or len(entries) < 100:
                    raise ValueError(f'{year}: incomplete country table')
                published_totals = {}
                for month, index in month_columns.items():
                    published = arrival_count(totals[index])
                    parsed = sum(entry['counts'][month] or 0 for entry in entries)
                    if published is None or parsed != published:
                        raise ValueError(f'{year}-{month:02d}: country sum {parsed} != total {published}')
                    published_totals[f'{year}-{month:02d}'] = published
                annual_total = arrival_count(totals[total_column])
                if annual_total is None or sum(entry['annual'] or 0 for entry in entries) != annual_total:
                    raise ValueError(f'{year}: country annual totals do not reconcile')
                identities = Counter(entry['country'] for entry in entries)
                unresolved = {'republic', 'equatorial', 'dominican', 'central african',
                              'congo', 'republic of zimbabwe'}
                records, excluded = [], []
                for entry in entries:
                    reasons = []
                    if identities[entry['country']] > 1 or normalise(entry['source']) in unresolved:
                        reasons.append('unresolved origin')
                    if normalise(entry['source']) == 'european union':
                        reasons.append('aggregate origin')
                    if entry['annual'] is not None and sum(v or 0 for v in entry['counts'].values()) != entry['annual']:
                        reasons.append('row total mismatch')
                    if reasons:
                        excluded.append({'country': entry['source'], 'row': entry['row'], 'reasons': reasons})
                        continue
                    for month, count in entry['counts'].items():
                        if count is not None:
                            records.append({'Year_Month': f'{year}-{month:02d}',
                                            'Country': entry['country'], 'Arrivals': count})
                return records, {'monthly_totals': published_totals, 'excluded_rows': excluded,
                                 'annual_month_difference': sum(published_totals.values()) - annual_total}
    finally:
        book.close()
    raise ValueError(f'{year}: country/month table not found')


def build_features(records):
    data = pd.DataFrame(records)
    if data.empty or data.duplicated(['Year_Month', 'Country']).any():
        raise ValueError('Empty data or duplicate country/month observations')
    data['Month'] = pd.PeriodIndex(data.Year_Month, freq='M')
    history = data[['Month', 'Country', 'Arrivals']]
    for lag in [1, 2, 3, 12]:
        earlier = history.copy()
        earlier.Month = earlier.Month + lag
        earlier = earlier.rename(columns={'Arrivals': f'lag_{lag}'})
        data = data.merge(earlier, on=['Month', 'Country'], how='left', validate='one_to_one')
    data['month_number'] = data.Month.dt.month
    # Exact month joins prevent a missing month from shifting the history.
    data = data.dropna().sort_values(['Month', 'Country'])[columns].reset_index(drop=True)
    for column in columns[2:]:
        data[column] = data[column].astype('int64')
    return data


def describe_changes(old, new):
    old = old.set_index(['Year_Month', 'Country'])
    new = new.set_index(['Year_Month', 'Country'])
    common = old.index.intersection(new.index)
    changed = old.loc[common].ne(new.loc[common]).any(axis=1)
    return {'added_rows': len(new.index.difference(old.index)),
            'removed_rows': len(old.index.difference(new.index)),
            'revised_rows': int(changed.sum()),
            'old_latest_month': old.index.get_level_values(0).max(),
            'latest_month': new.index.get_level_values(0).max()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-dir', type=Path, default=folder / '.sltda-downloads')
    parser.add_argument('--local-dir', type=Path, help='Use previously downloaded official_YEAR.xlsx files')
    parser.add_argument('--check-only', action='store_true', help='Validate without changing the dataset or manifest')
    args = parser.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'data/source_manifest.json'
    previous = json.loads(manifest_path.read_text())
    if args.local_dir:
        files = {item['year']: item['url'] for item in previous['files']}
    else:
        files = discover_files(download(source_page).decode('utf-8'))
    if not {item['year'] for item in previous['files']}.issubset(files):
        raise ValueError('A previously downloaded year is missing from SLTDA')
    records, sources, audits = [], [], {}
    for year, url in sorted(files.items()):
        content = ((args.local_dir / f'official_{year}.xlsx').read_bytes()
                   if args.local_dir else download(url))
        rows, audit = parse_workbook(content, year)
        records.extend(rows)
        audits[str(year)] = audit
        sources.append({'year': year, 'url': url, 'sha256': hashlib.sha256(content).hexdigest()})
        (args.download_dir / f'official_{year}.xlsx').write_bytes(content)
    data = build_features(records)
    existing = pd.read_csv(folder / 'data/Tourism_Arrivals.csv')
    if data.empty:
        raise ValueError('No complete-history rows remain after validation')
    changes = describe_changes(existing, data)
    if changes['latest_month'] < changes['old_latest_month']:
        raise ValueError('Source coverage has moved backwards; dataset left unchanged')
    published_latest = max(month for audit in audits.values() for month in audit['monthly_totals'])
    if changes['latest_month'] != published_latest:
        raise ValueError('Latest source month has no eligible rows')
    last_complete = str(pd.Period(datetime.now(timezone.utc).date(), freq='M') - 1)
    if published_latest > last_complete:
        raise ValueError('Source includes an unfinished or future month')
    changed = any(changes[key] for key in ['added_rows', 'removed_rows', 'revised_rows'])
    manifest = {'source_page': source_page, 'files': sources,
                'first_month': data.Year_Month.min(), 'latest_month': data.Year_Month.max(),
                'eligible_rows': len(data)}
    summary = (f"SLTDA data through {changes['latest_month']}: {len(data):,} eligible rows.\n\n"
               f"Added: {changes['added_rows']:,}; revised (including lag changes): {changes['revised_rows']:,}; "
               f"removed: {changes['removed_rows']:,}.\n\n"
               'Review source revisions and exclusions before merging. Historical evaluation results remain fixed.\n')
    (args.download_dir / 'summary.md').write_text(summary, encoding='utf-8')
    (args.download_dir / 'validation.json').write_text(json.dumps({'changes': changes, 'years': audits}, indent=2), encoding='utf-8')
    if not args.check_only:
        if changed:
            data.to_csv(folder / 'data/Tourism_Arrivals.csv', index=False, lineterminator='\n')
        if manifest != previous:
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(summary)
    print('Data changed.' if changed else 'No data changes.')


if __name__ == '__main__':
    main()