import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data_prep import load_and_merge, prepare_activity_df
from feature_engineering import (
    filter_invalid,
    map_categories,
    extract_time_features,
    encode_cyclical,
    clean_and_select,
    prepare_dataloaders,
)
from model import HierarchicalModel


def main():
    # File paths
    trip_fp = "BATS_2019_Trip.csv"
    person_fp = "BATS_2019_Person.csv"
    household_fp = "BATS_2019_Household.csv"

    # 1) Data preparation
    df = load_and_merge(trip_fp, person_fp, household_fp)
    act_df = prepare_activity_df(df)

    # 2) Feature engineering pipeline
    df1 = filter_invalid(act_df)
    df2 = map_categories(df1)
    df3 = extract_time_features(df2)
    df4 = encode_cyclical(df3)
    df_clean = clean_and_select(df4)

    # 3) Define columns
    cat_cols = [
        'activity_num',
        'day_of_week',
        'hour_of_day',
        'minute_of_hour',
        'relative_day',
    ]
    cont_cols = [
        'd_distance_home',
        'd_distance_work',
        'd_distance_school',
        'duration_min',
        'time_sin',
        'time_cos',
        'dow_sin',
        'dow_cos',
    ]
    target_cols = ['age_group', 'education_group', 'income_group']

    # 4) Create DataLoaders
    train_loader, val_loader = prepare_dataloaders(
        df_clean,
        cat_cols,
        cont_cols,
        target_cols,
        batch_size=32
    )

    # 5) Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cat_sizes = [df_clean[c].nunique() for c in cat_cols]
    num_cont = len(cont_cols)
    num_cls = {c: df_clean[c].nunique() for c in target_cols}

    model = HierarchicalModel(
        cat_sizes=cat_sizes,
        num_cont=num_cont,
        dim=64,
        seq_depth=6,
        seq_heads=16,
        num_cls=num_cls,
        seq_len=50,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    torch.backends.cudnn.benchmark = True

    # 6) Training loop
    epochs = 5
    for ep in range(1, epochs + 1):
        # --- Training ---
        model.train()
        total_loss = 0.0
        for (cats, conts), ys in train_loader:
            cats, conts = cats.to(device), conts.to(device)
            y_dict = {name: ys[:, i].to(device) for i, name in enumerate(target_cols)}

            optimizer.zero_grad()
            with autocast():
                outs = model(cats, conts)
                loss = sum(criterion(outs[name], y_dict[name]) for name in target_cols)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {ep:02d} | Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = {n: 0 for n in target_cols}
        total = 0
        with torch.no_grad():
            for (cats, conts), ys in val_loader:
                cats, conts = cats.to(device), conts.to(device)
                y_dict = {name: ys[:, i].to(device) for i, name in enumerate(target_cols)}
                with autocast():
                    outs = model(cats, conts)
                val_loss += sum(criterion(outs[name], y_dict[name]) for name in target_cols).item()
                preds = {name: outs[name].argmax(dim=1) for name in target_cols}
                for name in target_cols:
                    correct[name] += (preds[name] == y_dict[name]).sum().item()
                total += cats.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = {n: correct[n] / total for n in target_cols}
        print(f"          Val Loss: {avg_val_loss:.4f} | Accuracies: {acc}\n")

if __name__ == '__main__':
    main()
