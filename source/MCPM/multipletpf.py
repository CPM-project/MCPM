import numpy as np
from bisect import bisect
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from MCPM import tpfdata, hugetpf, utils
from MCPM.tpfgrid import TpfGrid
from MCPM.tpfrectangles import TpfRectangles


def limited_instances(class_):
    instances = {}
    def getinstance(campaign, channel, *args, **kwargs):
        key = tuple([int(campaign), int(channel)])
        if key not in instances:
            instances[key] = class_(campaign, channel, *args, **kwargs)
        return instances[key]
    return getinstance
# See also https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

# We want to make sure that for given campaign and channel there is only
# a single instance.
@limited_instances
class MultipleTpf(object):
    def __init__(self, campaign, channel):
        self.campaign = campaign
        self.channel = channel
        self.n_remove_huge = None

        self._tpfs = [] # This list has TpfData instances and the order 
        # corresponds to self._epic_ids.
        self._epic_ids = [] # This is sorted list and all elements are 
        # of string type.

        self._huge_tpf = None
        self._tpf_grid = None
        self._tpf_rectangles = None
        
        self._get_rows_columns_epics = None
        self._get_fluxes_epics = None
        self._get_median_fluxes_epics = None

    @property
    def campaign(self):
        """which campaign/subcampaign are we working on"""
        if self._campaign is None:
            raise ValueError("MultipleTpf.campaign not set")
        return self._campaign
        
    @campaign.setter
    def campaign(self, new_value):
        self._campaign = int(new_value)

    @property
    def channel(self):
        """which channel are we working on"""
        if self._channel is None:
            raise ValueError("MultipleTpf.channel not set")
        return self._channel
        
    @channel.setter
    def channel(self, new_value):
        self._channel = int(new_value)

    def add_tpf_data(self, tpf_data):
        """add one more instance of TpfData"""
        if not isinstance(tpf_data, tpfdata.TpfData):
            msg = ("Ooops... MultipleTpf.add_tpf_data() requires input that " +
                "is an instance of TpfData class; {:} given")
            raise ValueError(msg.format(type(tpf_data)))
        epic_id = str(tpf_data.epic_id)
        if epic_id in self._epic_ids:
            return
        if self.campaign is None:
            self.campaign = tpf_data.campaign
        if self.channel is None:
            self.channel = tpf_data.channel
        if not self._tpfs:
            if self.n_remove_huge is None:
                self._huge_tpf = hugetpf.HugeTpf(campaign=self.campaign)
            else:
                self._huge_tpf = hugetpf.HugeTpf(campaign=self.campaign, 
                                                    n_huge=self.n_remove_huge)
        else:
            if self.campaign != tpf_data.campaign:
                msg = ('MultipleTpf.add_tpf_data() cannot add data from ' + 
                        'a different campaign ({:} and {:})')
                raise ValueError(msg.format(self.campaign, tpf_data.campaign))
            if self.channel != tpf_data.channel:
                msg = ('MultipleTpf.add_tpf_data() cannot add data from ' + 
                        'a different channel ({:} and {:})')
                raise ValueError(msg.format(self.channel, tpf_data.channel))
        
        index = bisect(self._epic_ids, epic_id)
        self._tpfs.insert(index, tpf_data)
        self._epic_ids.insert(index, epic_id)
        
    def tpf_for_epic_id(self, epic_id, add_if_not_present=True):
        """returns an instance of TpfData corresponding to given epic_id"""
        epic_id = str(epic_id)
        if epic_id not in self._epic_ids:
            if not add_if_not_present:
                msg = 'EPIC {:} not in the MultipleTpf instance'
                raise ValueError(msg.format(epic_id))
            self.add_tpf_data_from_epic_list([epic_id])

        index = self._epic_ids.index(epic_id)
        return self._tpfs[index]  

    def add_tpf_data_from_epic_list(self, epic_id_list, campaign=None):
        """for each epic_id in the list, construct TPF object and add it"""
        if campaign is None:
            if self.campaign is None:
                raise ValueError('MultipleTpf - campaign not known')
            campaign = self.campaign
        for epic_id in epic_id_list:
            if str(epic_id) in self._epic_ids:
                continue
            if self._huge_tpf is not None:
                if str(epic_id) in self._huge_tpf.huge_ids:
                    continue
                    # This way we skip huge TPF files, though they can still 
                    # be added via self.add_tpf_data().
            new_tpf = tpfdata.TpfData(epic_id=epic_id, campaign=campaign)
            self.add_tpf_data(new_tpf)
        
    def _limit_epic_ids_to_list(self, epic_list):
        """limit self._epic_ids to ones in epic_list"""
        out = []
        for epic in self._epic_ids:
            if str(epic) in epic_list:
                out.append(str(epic))
        return out      
    
    def get_rows_columns(self, epics_to_include):
        """get concatenated rows and columns for selected epics"""
        get_rows_columns_epics = self._limit_epic_ids_to_list(epics_to_include)
        if get_rows_columns_epics == self._get_rows_columns_epics:
            return (self._get_rows_columns_rows, self._get_rows_columns_columns)

        rows = []
        columns = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            rows.append(self._tpfs[i].rows)
            columns.append(self._tpfs[i].columns)
        rows = np.concatenate(rows, axis=0).astype(int)
        columns = np.concatenate(columns, axis=0).astype(int)
        
        self._get_rows_columns_rows = rows
        self._get_rows_columns_columns = columns
        self._get_rows_columns_epics = get_rows_columns_epics
        return (self._get_rows_columns_rows, self._get_rows_columns_columns)

    def get_fluxes(self, epics_to_include):
        """get concatenated fluxes for selected epics"""
        get_fluxes_epics = self._limit_epic_ids_to_list(epics_to_include)
        if get_fluxes_epics == self._get_fluxes_epics:
            return self._get_fluxes

        flux = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            flux.append(self._tpfs[i].flux)
            
        self._get_fluxes = np.concatenate(flux, axis=1)
        self._get_fluxes_epics = get_fluxes_epics
        return self._get_fluxes
    
    def _get_epic_ids_as_vector(self, epics_to_include):
        """get vector that gives epic_id as many times as there are pixels"""
        get_fluxes_epics = self._limit_epic_ids_to_list(epics_to_include)

        epic_ids = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            epic_ids.extend([self._tpfs[i].epic_id] * self._tpfs[i].n_pixels)
            
        return np.array(epic_ids)
    
    def get_median_fluxes(self, epics_to_include):
        """get concatenated median fluxes for selected epics"""
        get_median_fluxes_epics = self._limit_epic_ids_to_list(epics_to_include)
        if get_median_fluxes_epics == self._get_median_fluxes_epics:
            return self._get_median_fluxes
        
        median_flux = []
        for (i, epic) in enumerate(self._epic_ids):
            if not epic in epics_to_include:
                continue
            median_flux.append(self._tpfs[i].median_flux)
            
        self._get_median_fluxes = np.concatenate(median_flux, axis=0)
        self._get_median_fluxes_epics = get_median_fluxes_epics
        return self._get_median_fluxes        

    def _mask_pixel_based_on_flux(self, target_row, target_column, 
            median_flux_ratio_limits, median_flux_limits, epics):
        """Prepare a pixel mask that is based on median fluxes"""
        if median_flux_ratio_limits is None and median_flux_limits is None:
            return np.array([True])
        ref_tpf = self.tpf_for_epic_id(epics[0])
        target_index = ref_tpf.get_pixel_index(row=target_row, column=target_column) 
        pixel_median = self.get_median_fluxes(epics)
        
        pixel_mask = None
        if median_flux_ratio_limits is not None: 
            if not (median_flux_ratio_limits[0] is None 
                    and median_flux_ratio_limits[1] is None):
                        
                ref_median_flux = ref_tpf.median_flux[target_index]
                lim_1 = median_flux_ratio_limits[0] * ref_median_flux
                lim_2 = median_flux_ratio_limits[1] * ref_median_flux
                mask_1 = (lim_1 <= pixel_median)
                mask_2 = (lim_2 >= pixel_median)
                pixel_mask = (mask_1 & mask_2)
            
        if median_flux_limits is not None:
            mask_1 = (median_flux_limits[0] <= pixel_median)
            mask_2 = (median_flux_limits[1] >= pixel_median)
            if pixel_mask is None:
                pixel_mask = (mask_1 & mask_2)
            else:
                pixel_mask &= (mask_1 & mask_2)
                
        return pixel_mask

    def _predictor_matrix_for_epics(self, x, y, n_pixel, min_distance, 
            exclude, median_flux_ratio_limits, median_flux_limits, 
            epics):
        """
        inner function that tries to get the predictor_matrix
        it requires epics - a list of TPF ids
        """
        target_column = int(x+0.5)
        target_row = int(y+0.5)
        self.add_tpf_data_from_epic_list(epics)
        
        (pixel_row, pixel_column) = self.get_rows_columns(epics)
        pixel_flux = self.get_fluxes(epics)
        pixel_mask = np.ones_like(pixel_row, dtype=bool)
        
        if exclude is not None: # exclude=None means no exclusion at all
            for shift in range(-exclude, exclude+1):
                pixel_mask &= (pixel_row != (target_row+shift))
                pixel_mask &= (pixel_column != (target_column+shift))

        pixel_mask &= self._mask_pixel_based_on_flux(target_row, target_column, 
            median_flux_ratio_limits, median_flux_limits, epics)
                
        distance2_row = np.square(pixel_row[pixel_mask] - target_row)
        distance2_column = np.square(pixel_column[pixel_mask] - target_column)
        distance2 = distance2_row + distance2_column
        distance_mask = (distance2 > min_distance**2)
        if np.sum(distance_mask) < n_pixel: # means we haven't found 
            return (None, None, np.sum(distance_mask)) # enough pixels
        distance2 = distance2[distance_mask]
        index = np.argsort(distance2, kind="mergesort")
        max_distance = distance2[index[n_pixel]]**.5

        pixel_numbers_masked = np.where(pixel_mask)[0]
        pixel_indexes = pixel_numbers_masked[distance_mask][index[:n_pixel]]
        predictor_flux = pixel_flux[:, pixel_indexes]
        predictor_row = pixel_row[pixel_indexes]
        predictor_column = pixel_column[pixel_indexes]

        used_epic_ids = set(self._get_epic_ids_as_vector(epics)[pixel_indexes])
        epoch_mask = np.ones(predictor_flux.shape[0], dtype=bool)
        for epic_id in used_epic_ids:
            epoch_mask &= self.tpf_for_epic_id(epic_id).epoch_mask
            
        return (predictor_flux, epoch_mask, max_distance, 
                predictor_row, predictor_column)
        
    def _guess_n_radius_min(self, n_pixel, exclude):
        """guess the minimum radius in which all pixels have to be checked"""
        margin = 1.2
        if exclude is None:
            n_rm = 0
        else:
            n_rm = 2 * exclude + 1
        delta = 4 * n_rm**2 + np.pi * (margin*n_pixel)
        radius_min = (2 * n_rm + (delta)**.5) / np.pi
        return radius_min
        
    @property
    def tpf_grid(self):
        """TpfGrid object"""
        if self._tpf_grid is None:
            self._tpf_grid = TpfGrid(self.campaign, self.channel)
        return self._tpf_grid
        
    @property
    def tpf_rectangles(self):
        """TpfRectangles object """
        if self._tpf_rectangles is None:
            self._tpf_rectangles = TpfRectangles(campaign=self.campaign, 
                                                        channel=self.channel)
        return self._tpf_rectangles
        
    def get_predictor_matrix(self, ra, dec, n_pixel=400, min_distance=15, 
            exclude=1, median_flux_ratio_limits=(None, None), 
            median_flux_limits=(100., 1.e5), n_pca_components=0):
        """Calculate predictor matrix.
        exclude - number or rows and columns around the target that would be
            rejected
        """
        n_add_epics = 3 # How many more epics we should consider in each loop

        (mean_x, mean_y) = self.tpf_grid.apply_grid_single(ra, dec)
        (epics, epics_distances) = self.tpf_rectangles.closest_epics(x=mean_x, y=mean_y)
        radius_min = self._guess_n_radius_min(n_pixel, exclude)
        n_epics = np.argmax(epics_distances>radius_min) + 2

        run = True
        while run:
            if n_epics > len(epics):
                raise ValueError('predictor_matrix preparation failed')
            out = self._predictor_matrix_for_epics(x=mean_x, y=mean_y,
                n_pixel=n_pixel, min_distance=min_distance, exclude=exclude,
                median_flux_ratio_limits=median_flux_ratio_limits,
                median_flux_limits=median_flux_limits,
                epics=epics[:n_epics])
            if out[0] is None:
                guess = n_epics * (n_pixel/float(out[2])) # Some rough guess of how 
                # many more epics we need.
                n_epics = max(n_epics + n_add_epics, int(guess+.5))
            elif out[2] > epics_distances[n_epics-2]:
                n_epics += n_add_epics
            else:
                run = False
        predictor_matrix = out[0]
        
        if n_pca_components > 0:
            #pca = PCA(n_components='mle', svd_solver='full')
            pca = PCA(n_components=n_pca_components, svd_solver='full')
            pca.fit(predictor_matrix)
            predictor_matrix = pca.transform(predictor_matrix)
            #print(predictor_matrix.shape)
            #print(pca.explained_variance_ratio_)
        out_0 = predictor_matrix
        return (out_0, out[1], out[3], out[4])

    def get_epic_id_for_radec(self, ra, dec):
        """find which tpf file given (ra,dec) belong to"""
        (mean_x, mean_y) = self.tpf_grid.apply_grid_single(ra, dec)
        return self.tpf_rectangles.get_epic_id_for_pixel(mean_x, mean_y)

    def get_quality_flags_for_pixels(self, pixels):
        """
        For every pixel extracts quality_flags from TPF and returns a list.
        Most probably all elements of the list are the same.
        """
        out = []
        for (x, y) in pixels:
            epic = self.tpf_rectangles.get_epic_id_for_pixel(x, y)
            # This could be faster - most probably current pixel is in
            # the same TPF as the previous one.
            out.append(self.tpf_for_epic_id(epic).quality_flags)
        return out

    def get_time_flux_mask_for_pixels(self, pixels):
        """
        for every pixel in the list extract the time, flux, and epoch masks
        from appropriate TPF files; pixels is a list of input pixels
        """
        time = []
        flux = []
        flux_err = []
        mask = []
        for (x, y) in pixels:
            epic = self.tpf_rectangles.get_epic_id_for_pixel(x, y)
            # This could be faster - most probably current pixel is in
            # the same TPF as the previous one.
            tpf = self.tpf_for_epic_id(epic)
            time.append(tpf.jd_short)
            flux.append(tpf.get_flux_for_pixel(row=y, column=x))
            flux_err.append(tpf.get_flux_err_for_pixel(row=y, column=x))
            mask.append(tpf.epoch_mask)
        return (time, flux, flux_err, mask)

    def plot_pixel_curves(self, mean_x=None, mean_y=None, half_size=2, 
            pixels=None, flux=None,
            figsize=(15, 10.3), dpi=300, point_size=2, time_mask=None, **kwargs):
        """
        Plot raw light curves for pixels in a square.
        Default settings produce large but readable file. 
        The **kwargs are passed to utils.plot_matrix_subplots().
        """
        fig_args = {"left":0.035, "bottom":0.023, "right":.995, "top":.995}

        if flux is not None and pixels is not None: # Data given directly. 
            if mean_x is not None or mean_y is not None:
                raise ValueError('error #1 in plot_pixel_curves()')
            flux_matrix = utils.construct_matrix_from_list(pixels, flux)
            epic = self.tpf_rectangles.get_epic_id_for_pixel(
                    pixels[0,0], pixels[0,1])
            tpf = self.tpf_for_epic_id(epic)
            time = tpf.jd_short
            if time_mask is not None:
                time = time[time_mask]
        # Data not given directly, it mean position specified:
        elif mean_x is not None and mean_y is not None and half_size is not None: 
            if flux is not None or pixels is not None:
                raise ValueError('error #2 in plot_pixel_curves()')
            pixels = utils.pixel_list_center(mean_x, mean_y, half_size)
            fluxes = []
            for (x, y) in pixels:
                epic = self.tpf_rectangles.get_epic_id_for_pixel(x, y)
                # This could be faster - most probably current pixel is in
                # the same TPF as the previous one.
                tpf = self.tpf_for_epic_id(epic)
                fluxes.append(tpf.get_flux_for_pixel(row=y, column=x))
            flux_matrix = utils.construct_matrix_from_list(pixels, fluxes)
            time = tpf.jd_short
            if time_mask is not None:
                time = time[time_mask]
        else:
            raise ValueError('error #3 in plot_pixel_curves()')
        
        flux_matrix[flux_matrix == 0.] = None

        # Plotting starts here.
        (fig, ax) = plt.subplots(figsize=figsize, dpi=dpi)
        utils.plot_matrix_subplots(fig, time, flux_matrix, 
            ms=point_size, **kwargs)
        plt.subplots_adjust(**fig_args)
