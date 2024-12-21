from ast import literal_eval
import pandas as pd
from math import sin, radians, cos, sqrt, atan2

pd.options.mode.chained_assignment = None


class DataGenerationModule:
    def __init__(
        self,
        courier_wave_path,
        all_waybill_path,
        dispatching_order_path,
        dispatch_waybill_path,
    ):
        # Load datasets
        self.courier_wave_df = pd.read_csv(courier_wave_path)
        self.all_waybill_df = pd.read_csv(all_waybill_path)
        self.dispatching_order_df = pd.read_csv(dispatching_order_path)
        self.dispatch_waybill_df = pd.read_csv(dispatch_waybill_path)

        # Preprocess datasets
        self.preprocess_data()

    def preprocess_data(self):
        # Drop redundant columns
        self.courier_wave_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        self.all_waybill_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        self.dispatching_order_df.drop(
            columns=["Unnamed: 0"], inplace=True, errors="ignore"
        )
        self.dispatch_waybill_df.drop(
            columns=["Unnamed: 0"], inplace=True, errors="ignore"
        )

        self.all_waybill_df["grab_time"] = pd.to_datetime(
            self.all_waybill_df["grab_time"], unit="s"
        )
        # Convert string representations of lists to Python lists
        if "order_ids" in self.courier_wave_df.columns:
            self.courier_wave_df["order_ids"] = self.courier_wave_df["order_ids"].apply(
                literal_eval
            )
        if "courier_waybills" in self.dispatching_order_df.columns:
            self.dispatching_order_df["courier_waybills"] = self.dispatching_order_df[
                "courier_waybills"
            ].apply(literal_eval)

        # Convert timestamps to datetime where applicable
        time_columns = [
            "wave_start_time",
            "wave_end_time",
            "dispatch_time",
            "fetch_time",
            "arrive_time",
        ]
        for col in time_columns:
            if col in self.all_waybill_df.columns:
                self.all_waybill_df[col] = pd.to_datetime(
                    self.all_waybill_df[col], unit="s"
                )
            if col in self.courier_wave_df.columns:
                self.courier_wave_df[col] = pd.to_datetime(
                    self.courier_wave_df[col], unit="s"
                )

        # Add a unique key for merging (courier_id and date)
        self.all_waybill_df["courier_dt"] = (
            self.all_waybill_df["courier_id"].astype(str)
            + "_"
            + self.all_waybill_df["dt"].astype(str)
        )
        self.courier_wave_df["courier_dt"] = (
            self.courier_wave_df["courier_id"].astype(str)
            + "_"
            + self.courier_wave_df["dt"].astype(str)
        )

        # Merge with specified suffixes to avoid column conflict
        merged_df = pd.merge(
            self.all_waybill_df,
            self.courier_wave_df,
            on=["courier_dt"],
            suffixes=("_order", "_wave"),  # Differentiates columns with same name
            how="inner",  # Ensures only matching rows are included
        )

        # Ensure we are referencing the correct columns
        dispatch_column = "dispatch_time"  # Comes from all_waybill_df
        courier_id_column = "courier_id_order"  # courier_id from all_waybill_df
        wave_start_column = "wave_start_time"  # courier_wave_df
        wave_end_column = "wave_end_time"  # courier_wave_df

        # Filter rows where dispatch_time is within the wave duration
        merged_df = merged_df[
            (merged_df[dispatch_column] >= merged_df[wave_start_column])
            & (merged_df[dispatch_column] <= merged_df[wave_end_column])
        ]

        # Count active orders from the order_ids field
        merged_df["active_orders"] = merged_df["order_ids"].apply(
            lambda x: len(eval(x)) if isinstance(x, str) else 0
        )

        # Keep only the necessary columns
        result_df = merged_df[
            [dispatch_column, courier_id_column, "active_orders"]
        ].drop_duplicates()

        # Merge active_orders back into the original all_waybill_df
        order = pd.merge(
            self.all_waybill_df,
            result_df,
            left_on=["courier_id", "dispatch_time"],
            right_on=[courier_id_column, dispatch_column],
            how="left",
        )

        # Fill NaN values in active_orders with 0
        order["active_orders"] = order["active_orders"].fillna(0).astype(int)

        rejection_rate = (
            order.groupby("courier_id")["is_courier_grabbed"].mean().reset_index()
        )
        rejection_rate.columns = ["courier_id", "historical_acceptance_rate"]
        order = pd.merge(order, rejection_rate, on="courier_id", how="left")
        order["historical_rejection_rate"] = 1 - order["historical_acceptance_rate"]

        order["hour_of_day"] = pd.to_datetime(
            order["platform_order_time"], unit="s"
        ).dt.hour

        order["peak_hours"] = order["hour_of_day"].apply(
            lambda x: 1 if 11 <= x <= 13 or 18 <= x <= 20 else 0
        )

        # Count active orders in the same area (da_id)
        area_orders = (
            order.groupby(["da_id", "dispatch_time"])["order_id"].count().reset_index()
        )
        area_orders.columns = ["da_id", "dispatch_time", "active_area_orders"]

        # Merge back into the main dataframe
        order = pd.merge(order, area_orders, on=["da_id", "dispatch_time"], how="left")

        # Fill NaN values for new features
        order["active_area_orders"].fillna(0, inplace=True)

        # Convert dispatch_time to datetime if not already
        order["dispatch_time"] = pd.to_datetime(order["dispatch_time"], unit="s")

        # Get the last dispatch time per courier per day
        courier_last_dispatch = (
            order.groupby(["courier_id", "dt"])["dispatch_time"].max().reset_index()
        )
        courier_last_dispatch.columns = ["courier_id", "dt", "last_dispatch_time"]

        # Merge last dispatch time back into the main DataFrame
        order = pd.merge(
            order, courier_last_dispatch, on=["courier_id", "dt"], how="left"
        )

        # Calculate time difference to the last dispatch time
        order["time_to_shift_end"] = (
            order["last_dispatch_time"] - order["dispatch_time"]
        ).dt.total_seconds()

        # Flag dispatches occurring near the shift end (e.g., within 30 minutes)
        order["near_shift_end"] = order["time_to_shift_end"].apply(
            lambda x: 1 if 0 <= x <= 1800 else 0  # 1800 seconds = 30 minutes
        )

        self.all_waybill_df = order.copy()

    def get_active_couriers(self, timestamp):
        """
        Retrieve active couriers at the given timestamp.
        """
        timestamp = pd.to_datetime(timestamp, unit="s")
        active_couriers = self.courier_wave_df[
            (self.courier_wave_df["wave_start_time"] <= timestamp)
            & (self.courier_wave_df["wave_end_time"] >= timestamp)
        ]
        # Retrieve the most recent location of each courier from all_waybill_df
        courier_locations = (
            self.all_waybill_df[self.all_waybill_df["grab_time"] <= timestamp]
            .sort_values(by="grab_time", ascending=False)
            .drop_duplicates(subset=["courier_id"])[
                ["courier_id", "grab_lat", "grab_lng"]
            ]
        )
        # Merge active couriers with locations
        active_couriers = active_couriers.merge(
            courier_locations, on="courier_id", how="left"
        )
        # Fill missing locations and details with default values
        active_couriers["grab_lat"] = active_couriers["grab_lat"].fillna(0)
        active_couriers["grab_lng"] = active_couriers["grab_lng"].fillna(0)
        active_couriers["wave_id"] = active_couriers["wave_id"].fillna(-1)
        active_couriers["order_ids"] = active_couriers["order_ids"].fillna("[]")
        # Add this at the end of the get_active_couriers method
        active_couriers["unfulfilled_orders"] = 0  # Initialize with 0
        return active_couriers[
            [
                "courier_id",
                "wave_id",
                "order_ids",
                "grab_lat",
                "grab_lng",
                "unfulfilled_orders",
            ]
        ]

    def get_unfulfilled_orders(self, courier_order_ids, timestamp):
        """
        Retrieve the number of unfulfilled orders for a courier before the given timestamp.
        """
        if not courier_order_ids:  # Handle case where no orders are assigned
            return 0

        timestamp = pd.to_datetime(timestamp, unit="s")

        # Filter orders assigned to the courier
        unfulfilled_orders = self.all_waybill_df[
            (self.all_waybill_df["order_id"].isin(courier_order_ids))
            & (self.all_waybill_df["arrive_time"] > timestamp)
        ]

        # Return the count of unfulfilled orders
        return len(unfulfilled_orders)

    def get_orders_in_time_window(self, start_time, end_time):
        start_time = pd.to_datetime(start_time, unit="s")
        end_time = pd.to_datetime(end_time, unit="s")
        orders_in_window = self.all_waybill_df[
            (self.all_waybill_df["dispatch_time"] >= start_time)
            & (self.all_waybill_df["dispatch_time"] < end_time)
        ]
        return orders_in_window[
            [
                "order_id",
                "sender_lat",
                "sender_lng",
                "estimate_meal_prepare_time",
                "order_push_time",
            ]
        ]

    def construct_state(self, timestamp):
        """
        Construct the RL environment state at the given timestamp.
        """
        # Get active orders
        active_orders = self.get_active_orders(timestamp)

        # Get active couriers and count their unfulfilled orders
        active_couriers = self.get_active_couriers(timestamp)
        courier_details = []
        for _, row in active_couriers.iterrows():
            unfulfilled_count = self.get_unfulfilled_orders(row["order_ids"], timestamp)
            courier_details.append(
                {
                    "courier_id": row["courier_id"],
                    "wave_id": row["wave_id"],
                    "grab_lat": row["grab_lat"],
                    "grab_lng": row["grab_lng"],
                    "unfulfilled_orders": unfulfilled_count,
                }
            )

        # Construct state dictionary
        state = {
            "orders": active_orders.to_dict(orient="records"),
            "couriers": courier_details,
            "system": {
                "active_orders": len(active_orders),
                "active_couriers": len(active_couriers),
            },
        }
        return state


# Example usage
if __name__ == "__main__":
    # File paths
    courier_wave_path = "courier_wave_info_meituan.csv"
    all_waybill_path = "all_waybill_info_meituan_0322.csv"
    dispatching_order_path = "dispatch_rider_meituan.csv"
    dispatch_waybill_path = "dispatch_waybill_meituan.csv"

    # Initialize the data generation module
    data_module = DataGenerationModule(
        courier_wave_path,
        all_waybill_path,
        dispatching_order_path,
        dispatch_waybill_path,
    )

    # Define the timestamp
    timestamp = 1666077600  # Example timestamp (Unix time)

    # Construct state and save to JSON
    # state = data_module.construct_state(timestamp)

    orders_in_window = data_module.get_orders_in_time_window(
        1666077600, 1666077600 + 3600
    )
    print(orders_in_window)
    # with open("state.json", "w") as f:
    #     json.dump(state, f, indent=4)

    # print("State saved to 'state.json'")
    # print(json.dumps(state, indent=4))
