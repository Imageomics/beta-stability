import math
import pandas as pd


import numpy as np
import time
import logging
import contextlib

logger = logging.getLogger("beta_stability")


class MetadataEmbeddings:
    """
    Embeddings wrapper that filters based on geographic/temporal metadata.
    Returns 0.0 for implausible matches, original score for plausible matches.
    """

    def __init__(self, base_embeddings, data_df, node2uuid, id_key='uuid'):
        self.base_embeddings = base_embeddings
        self.metadata_filter = metadata_verifier(data_df, node2uuid, id_key)

        logger.info(f"Created MetadataEmbeddings wrapping {type(base_embeddings).__name__}")

    def _is_plausible(self, n0, n1):
        """Check if edge is plausible based on metadata."""
        plausible_pairs = self.metadata_filter([(n0, n1)])
        return len(plausible_pairs) > 0

    def get_score(self, n0, n1):
        """Return 0.0 if implausible, base score if plausible."""
        if self._is_plausible(n0, n1):
            return self.base_embeddings.get_score(n0, n1)
        return 0.0

    def _apply_metadata_filter_to_distmat(self, distmat, ids, start_idx):
        """Apply metadata filtering to distance matrix by setting implausible edges to infinity."""
        for i in range(distmat.shape[0]):
            for j in range(distmat.shape[1]):
                if i + start_idx != j:  # Don't filter self-edges
                    id1, id2 = ids[i + start_idx], ids[j]
                    if not self._is_plausible(id1, id2):
                        distmat[i, j] = 1
        return distmat

    def get_edges(self, topk=5, target_edges=10000, target_proportion=None, uuids_filter=None):
        """
        Get edges filtered by metadata plausibility.
        Applies metadata filtering to distance matrix BEFORE top-k selection.
        """
        start_time = time.time()
        print("Calculating distances with metadata filtering...")

        # Get distance matrix chunks from base embeddings
        chunks, ids = self.base_embeddings.get_distmat_chunks(uuids_filter=uuids_filter)
        result = []
        start = 0

        if target_proportion is None:
            embeds_num = len(ids)
            total_edges = (embeds_num * embeds_num - embeds_num) / 2
            target_proportion = np.clip(target_edges / total_edges, 0, 1)
        else:
            target_edges = int(total_edges * target_proportion)

        print(f"Target: {target_edges}/{total_edges}")

        for distmat in chunks:
            # FIRST: Apply metadata filtering to distance matrix
            distmat = self._apply_metadata_filter_to_distmat(distmat, ids, start)

            # THEN: Apply top-k selection on filtered distance matrix
            sorted_dists = distmat.argsort(axis=1).argsort(axis=1) < topk

            all_inds_y, all_inds_x = np.triu_indices(n=distmat.shape[0], m=distmat.shape[1], k=start+1)
            chunk_len = len(all_inds_y)
            order = np.random.permutation(chunk_len)[:int(chunk_len * target_proportion)]
            all_inds_y = all_inds_y[order]
            all_inds_x = all_inds_x[order]

            selected_dists = np.full(distmat.shape, False)
            selected_dists[all_inds_y, all_inds_x] = True

            filtered = np.logical_or(sorted_dists, selected_dists)
            inds_y, inds_x = np.nonzero(filtered)

            # Create edges from filtered distance matrix
            for (ind1, ind2) in zip(inds_y, inds_x, strict=True):
                score = 1 - distmat[ind1, ind2]
                if score > 0 and not np.isinf(score):  # Only valid edges
                    id1, id2 = ids[ind1 + start], ids[ind2]
                    result.append((*sorted([id1, id2]), score))

            print(f"Chunk result: {time.time() - start_time:.6f} seconds")
            start_time = time.time()
            start += distmat.shape[0]

        print(f"Calculated distances: {time.time() - start_time:.6f} seconds")
        print(f"Metadata filtered edges: {len(set(result))}")
        return set(result)

    def __getattr__(self, name):
        """Delegate other methods to base embeddings."""
        return getattr(self.base_embeddings, name)

    def get_base(self):
        """Return the unwrapped base embeddings for threshold computation."""
        return self.base_embeddings

    def get_stats(self, df, filter_key, id_key='uuid'):
        """Get statistics with metadata filtering applied."""
        print("Calculating distances with metadata filtering...")
        print(f"{len(self.base_embeddings.embeddings)}/{len(self.base_embeddings.ids)}")

        # Get distance matrix chunks from base embeddings
        chunks, ids = self.base_embeddings.get_distmat_chunks()
        filtered_chunks = []
        start = 0

        # Apply metadata filtering to each chunk
        for distmat in chunks:
            distmat = self._apply_metadata_filter_to_distmat(distmat, ids, start)
            filtered_chunks.append(distmat)
            start += distmat.shape[0]

        # Concatenate filtered chunks
        distmat = np.concatenate(filtered_chunks, axis=0)

        # Get labels and compute top-k statistics on filtered matrix
        labels = [df.loc[df[id_key] == self.base_embeddings.uuids[id], filter_key].values[0] for id in ids]
        top1, top3, top5, top10 = self.base_embeddings.get_top_ks(labels, distmat, ks=[1, 3, 5, 10])

        return top1, top3, top5, top10

class metadata_verifier(object):
    def __init__(self, data_df, node2uuid, id_key='uuid'):
        self.node2uuid = node2uuid
        self.uuid2node = {val:key for (key, val) in node2uuid.items()}
        self.id_key = id_key

        # Detect available column names (support multiple naming conventions)
        cols = set(data_df.columns)
        datetime_col = 'datetime' if 'datetime' in cols else 'timestamp'
        filename_col = 'file_name' if 'file_name' in cols else 'image_path'

        logger.info(f"metadata_verifier using datetime_col='{datetime_col}', filename_col='{filename_col}'")

        # Pre-cache all metadata: uuid -> (lon, lat, datetime_parsed, filename)
        # This avoids O(n) DataFrame scans per lookup
        self._cache = {}
        has_gps = 'gps_lon' in cols and 'gps_lat' in cols
        has_datetime = datetime_col in cols
        has_filename = filename_col in cols

        for _, row in data_df.iterrows():
            uuid = row[id_key]

            # Parse GPS
            lon, lat = -1.0, -1.0
            if has_gps:
                try:
                    lon = float(row['gps_lon'])
                    lat = float(row['gps_lat'])
                except (ValueError, TypeError):
                    pass

            # Parse datetime. Numeric values are Unix epoch seconds; without
            # unit='s' pandas reads them as nanoseconds, which puts every image
            # within seconds of 1970-01-01 and makes all pairs look simultaneous.
            dt_parsed = None
            if has_datetime:
                with contextlib.suppress(Exception):
                    value = row[datetime_col]
                    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                        dt_parsed = pd.to_datetime(value, unit='s')
                    else:
                        dt_parsed = pd.to_datetime(value)

            # Get filename
            filename = str(row[filename_col]) if has_filename else ""

            self._cache[uuid] = (lon, lat, dt_parsed, filename)

        logger.info(f"metadata_verifier cached {len(self._cache)} annotations")

    def _get_cached(self, uuid):
        """Get pre-cached metadata for a uuid. Returns (lon, lat, datetime, filename)."""
        return self._cache.get(uuid, (-1.0, -1.0, None, ""))

    def __call__(self, query):
        nodes_to_review = []

        MAX_ZEBRA_SPEED_KMH = 65
        MAX_DISTANCE_KM = 70
        rejected_count = 0

        for (n0, n1) in query:
            uuid1 = self.node2uuid[n0]
            uuid2 = self.node2uuid[n1]
            lon1, lat1, dt1, file1 = self._get_cached(uuid1)
            lon2, lat2, dt2, file2 = self._get_cached(uuid2)

            if file1 == file2:
                logger.debug(f"Metadata REJECTED ({uuid1[:8]}, {uuid2[:8]}): same image file '{file1}'")
                rejected_count += 1
                continue

            # Missing datetime → plausible
            if dt1 is None or dt2 is None:
                nodes_to_review.append((uuid1, uuid2))
                continue

            time_diff_hours = abs((dt1 - dt2).total_seconds() / 3600)

            # Missing GPS → plausible
            if lon1 == -1 or lat1 == -1 or lon2 == -1 or lat2 == -1:
                nodes_to_review.append((uuid1, uuid2))
                continue

            # Haversine distance calculation
            R = 6371
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)
            a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = R * c

            # Reject pairs that are too far apart regardless of time
            if distance_km > MAX_DISTANCE_KM:
                logger.debug(f"Metadata REJECTED ({uuid1[:8]}, {uuid2[:8]}): {distance_km:.1f}km apart (max {MAX_DISTANCE_KM}km)")
                rejected_count += 1
                continue

            # Avoid division by zero for same timestamp
            if time_diff_hours <= 0.1:
                plausible = distance_km < 0.1  # 100 meters, basically the same spot
                if not plausible:
                    logger.debug(f"Metadata REJECTED ({uuid1[:8]}, {uuid2[:8]}): same time (<0.1h) but {distance_km:.2f}km apart (max 0.1km)")
            else:
                required_speed = distance_km / time_diff_hours
                plausible = required_speed <= MAX_ZEBRA_SPEED_KMH
                if not plausible:
                    logger.debug(f"Metadata REJECTED ({uuid1[:8]}, {uuid2[:8]}): {distance_km:.1f}km in {time_diff_hours:.1f}h = {required_speed:.1f}km/h (max {MAX_ZEBRA_SPEED_KMH}km/h)")

            if plausible:
                nodes_to_review.append((uuid1, uuid2))
            else:
                rejected_count += 1

        return nodes_to_review
